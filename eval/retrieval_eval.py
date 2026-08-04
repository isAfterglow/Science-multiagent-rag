"""Reproducible retrieval ablation with source-level human relevance labels."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import quantiles

from app.config import DB_PATH, ROOT
from app.registry import SimulationRegistry
from app.retrieval import HybridRetriever

EVAL_PATH = Path(__file__).with_name("retrieval_ground_truth.jsonl")


def _questions() -> list[dict]:
    return [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_name(path: str) -> str:
    return Path(path).name


def _metrics(rows: list[dict]) -> dict:
    total = len(rows)
    reciprocal_ranks = []
    ndcgs = []
    for row in rows:
        relevant = set(row["relevant_sources"])
        seen: set[str] = set()
        hits = []
        for index, source in enumerate(row["returned_sources"], 1):
            if source in relevant and source not in seen:
                hits.append(index)
                seen.add(source)
        reciprocal_ranks.append(1 / hits[0] if hits else 0)
        dcg = sum(1 / math.log2(rank + 1) for rank in hits)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(5, len(relevant)) + 1))
        ndcgs.append(dcg / ideal if ideal else 0)
    latencies = [row["latency_ms"] for row in rows]
    p95 = quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0)
    return {
        "questions": total,
        "recall_at_5": round(sum(bool(row["hit"]) for row in rows) / total, 4),
        "mrr": round(sum(reciprocal_ranks) / total, 4),
        "ndcg_at_5": round(sum(ndcgs) / total, 4),
        "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else 0,
        "p95_latency_ms": round(p95, 3),
    }


def evaluate(mode: str, chunk_strategy: str, registry: SimulationRegistry, max_questions: int | None = None) -> dict:
    retriever = HybridRetriever(registry, mode=mode, chunk_strategy=chunk_strategy)
    rows = []
    for item in _questions()[:max_questions]:
        started = time.perf_counter()
        cards = retriever.search(item["question"], limit=5)
        elapsed = (time.perf_counter() - started) * 1000
        sources = [_source_name(card.source_path) for card in cards]
        rows.append({
            "id": item["id"], "question": item["question"], "relevant_sources": item["relevant_sources"],
            "returned_sources": sources, "returned_chunk_ids": [card.retrieval.get("chunk_id") for card in cards],
            "hit": bool(set(sources) & set(item["relevant_sources"])), "latency_ms": round(elapsed, 3),
        })
    return {"mode": mode, "chunk_strategy": chunk_strategy, "chunk_count": len(retriever.chunks), "metrics": _metrics(rows), "results": rows}


def run_all() -> dict:
    registry = SimulationRegistry(DB_PATH)
    chunking = [evaluate("hybrid", strategy, registry) for strategy in ("document", "fixed", "structure", "parent_child")]
    best = max(chunking, key=lambda row: (row["metrics"]["ndcg_at_5"], row["metrics"]["recall_at_5"], -row["metrics"]["p95_latency_ms"]))
    strategies = [evaluate(mode, best["chunk_strategy"], registry) for mode in ("bm25", "dense", "hybrid", "hybrid_rerank")]
    return {"ground_truth": str(EVAL_PATH), "chunking_ablation": chunking, "strategy_ablation": strategies, "recommended_chunk_strategy": best["chunk_strategy"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
    parser.add_argument("--chunk-strategy", choices=["document", "fixed", "structure", "parent_child"])
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "retrieval_ablation.json")
    parser.add_argument("--max-questions", type=int)
    args = parser.parse_args()
    registry = SimulationRegistry(DB_PATH)
    result = evaluate(args.mode, args.chunk_strategy, registry, args.max_questions) if args.mode else run_all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
