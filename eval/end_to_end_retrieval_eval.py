"""Compare retrieval configurations inside the evidence-constrained graph."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import quantiles

from app.config import DB_PATH, ROOT
from app.llm_router import LLMRouter
from app.multi_agent import run_multi_agent
from app.registry import SimulationRegistry


def _questions() -> list[dict]:
    path = Path(__file__).with_name("stage2_questions.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(mode: str, chunk_strategy: str = "parent_child") -> dict:
    registry = SimulationRegistry(DB_PATH)
    router = LLMRouter(enabled=False)
    rows = []
    for item in _questions():
        started = time.perf_counter()
        result = run_multi_agent(item["question"], registry, router, mode, chunk_strategy)
        elapsed = (time.perf_counter() - started) * 1000
        citations = [Path(card["source_path"]).name for card in result["evidence_cards"]]
        rows.append({"id": item["id"], "approved": bool(result["review"].get("approved")), "has_analysis": bool(result["analysis_evidence"]), "has_citation": bool(citations), "citations": citations, "latency_ms": round(elapsed, 3)})
    latencies = [row["latency_ms"] for row in rows]
    return {"mode": mode, "chunk_strategy": chunk_strategy, "questions": len(rows), "review_pass_rate": round(sum(row["approved"] for row in rows) / len(rows), 4), "dual_evidence_rate": round(sum(row["has_analysis"] and row["has_citation"] for row in rows) / len(rows), 4), "p95_latency_ms": round(quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies), 3), "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    parser.add_argument("--chunk-strategy", default="parent_child")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "end_to_end_retrieval_comparison.json")
    args = parser.parse_args()
    result = [evaluate(mode, args.chunk_strategy) for mode in args.modes]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
