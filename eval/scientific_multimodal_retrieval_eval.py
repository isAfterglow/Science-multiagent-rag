"""Page-aware evaluation for the scientific multimodal corpus."""
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

EVAL_PATH = Path(__file__).with_name("scientific_multimodal_questions.jsonl")


def _questions() -> list[dict]:
    return [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_id(card: dict) -> str:
    metadata = card.get("retrieval", {})
    if metadata.get("source_id"):
        return metadata["source_id"]
    path = Path(card["source_path"]).name
    mapping = {
        "layer1_lhs_analysis_summary.txt": "project_lhs_report", "batch_run_layer1_lhs.log": "project_run_log",
        "run_batch_layer1_cases.py": "project_run_script", "case1_fiat_walltemp_nominal.i": "project_input_deck",
        "batch_run_status_layer1_lhs.csv": "project_run_status", "generate_lhs_layer1_main_cases.py": "project_lhs_script",
    }
    return mapping.get(path, path)


def question_tags(item: dict) -> list[str]:
    """Reusable query-shape slices, independent of question IDs or answers."""
    query = item["question"].lower()
    tags = ["ocr_scan"] if item.get("requires_ocr") else []
    if any(term in query for term in ("表格", "table", "单位", "变量", "equation", "mole fractions", "编号", "专利号")):
        tags.append("exact_table_term")
    if any(term in query for term in ("机理", "为什么", "影响", "局限", "驱动力", "目的", "性能目标")):
        tags.append("mechanism_explanation")
    if any(term in query for term in ("moose", "fiat", "输入 deck", "并行", "thermal hydraulics")):
        tags.append("moose_method")
    if len(item.get("relevant_source_ids", [])) > 1 or any(term in query for term in ("比较", "结合", "和外部", "多个来源")):
        tags.append("cross_source")
    return tags or ["general"]


def _summary(rows: list[dict]) -> dict:
    latencies = [row["latency_ms"] for row in rows]
    reciprocal = [1 / row["first_hit_rank"] if row["first_hit_rank"] else 0 for row in rows]
    dcg = [sum(1 / math.log2(rank + 1) for rank in row["hit_ranks"]) for row in rows]
    ideal = [sum(1 / math.log2(rank + 1) for rank in range(1, min(5, len(row["expected_sources"])) + 1)) for row in rows]
    scientific = [row for row in rows if any(source.startswith("nasa_") for source in row["expected_sources"])]
    ocr = [row for row in rows if row["requires_ocr"]]
    block_rows = [row for row in rows if row["expected_block_types"]]
    kind_rows = [row for row in rows if row.get("expected_document_kind")]
    def rate(items: list[dict], key: str) -> float:
        return round(sum(bool(item[key]) for item in items) / len(items), 4) if items else 0.0
    source_recall = [len(set(row["returned_sources"]) & set(row["expected_sources"])) / len(row["expected_sources"]) for row in rows]
    summary = {"questions": len(rows), "source_recall_at_5": round(sum(source_recall) / len(rows), 4), "question_hit_rate": rate(rows, "source_hit"), "mrr": round(sum(reciprocal) / len(rows), 4),
            "ndcg_at_5": round(sum(value / norm if norm else 0 for value, norm in zip(dcg, ideal)) / len(rows), 4),
            "page_hit_rate": rate([row for row in rows if row["expected_pages"]], "page_hit"),
            "authority_correct_rate": rate(rows, "authority_hit"), "scientific_source_recall_at_5": rate(scientific, "source_hit"),
            "ocr_source_recall_at_5": rate(ocr, "source_hit"), "ocr_page_hit_rate": rate(ocr, "page_hit"),
            "block_type_hit_rate": rate(block_rows, "block_type_hit"),
            "document_kind_hit_rate": rate(kind_rows, "document_kind_hit"),
            "p50_latency_ms": round(sorted(latencies)[len(latencies)//2], 3), "p95_latency_ms": round(quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies), 3)}
    subsets = {}
    for tag in sorted({tag for row in rows for tag in row.get("tags", [])}):
        selected = [row for row in rows if tag in row.get("tags", [])]
        subsets[tag] = {"questions": len(selected), "source_hit_rate": rate(selected, "source_hit"), "page_hit_rate": rate([row for row in selected if row["expected_pages"]], "page_hit"), "mrr": round(sum(1 / row["first_hit_rank"] if row["first_hit_rank"] else 0 for row in selected) / len(selected), 4)}
    summary["subsets"] = subsets
    return summary


def evaluate(mode: str, strategy: str, limit: int = 5, vector_backend: str | None = None) -> dict:
    retriever = HybridRetriever(SimulationRegistry(DB_PATH), mode=mode, chunk_strategy=strategy, vector_backend=vector_backend)
    rows: list[dict] = []
    for item in _questions():
        started = time.perf_counter(); cards = [card.model_dump() for card in retriever.search(item["question"], limit=limit)]; elapsed = (time.perf_counter() - started) * 1000
        returned_sources = [_source_id(card) for card in cards]
        returned_pages = [card["retrieval"].get("page") for card in cards]
        returned_block_types = [card["retrieval"].get("block_type") for card in cards]
        returned_document_kinds = [card["retrieval"].get("document_kind", card["source_type"]) for card in cards]
        expected = set(item["relevant_source_ids"]); expected_pages = set(item["relevant_pages"])
        seen: set[str] = set()
        hit_ranks = [index for index, source in enumerate(returned_sources, 1) if source in expected and not (source in seen or seen.add(source))]
        authorities = {card["retrieval"].get("authority") for card in cards}
        rows.append({"id": item["id"], "question": item["question"], "tags": question_tags(item), "expected_sources": sorted(expected), "expected_pages": sorted(expected_pages),
                     "returned_sources": returned_sources, "returned_pages": returned_pages, "source_hit": bool(hit_ranks), "first_hit_rank": hit_ranks[0] if hit_ranks else None,
                     "hit_ranks": hit_ranks, "page_hit": bool(expected_pages & set(page for page in returned_pages if page is not None)),
                     "authority_hit": set(item["required_authority"]).issubset(authorities), "requires_ocr": item["requires_ocr"],
                     "expected_block_types": item.get("expected_block_types", []), "returned_block_types": returned_block_types,
                     "block_type_hit": bool(set(item.get("expected_block_types", [])) & set(kind for kind in returned_block_types if kind)),
                     "expected_document_kind": item.get("expected_document_kind"), "returned_document_kinds": returned_document_kinds,
                     "document_kind_hit": not item.get("expected_document_kind") or item["expected_document_kind"] in returned_document_kinds,
                     "latency_ms": round(elapsed, 3)})
    return {"mode": mode, "chunk_strategy": strategy, "vector_backend": retriever.vector_status(), "chunk_count": len(retriever.chunks), "metrics": _summary(rows), "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "dense_page", "hybrid", "hybrid_rerank", "source_fusion"]); parser.add_argument("--chunk-strategy", default="parent_child"); parser.add_argument("--vector-backend", choices=["local", "milvus"], default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "scientific_multimodal_retrieval_eval.json")
    args = parser.parse_args(); result = evaluate(args.mode, args.chunk_strategy, vector_backend=args.vector_backend); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
