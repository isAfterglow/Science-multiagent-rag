"""Answer-level regression for non-template scientific workflow questions."""
from __future__ import annotations

import json
import argparse
import time
from pathlib import Path
from statistics import quantiles

from app.config import DB_PATH, ROOT
from app.llm_router import LLMRouter
from app.multi_agent import run_multi_agent
from app.registry import SimulationRegistry

QUESTIONS = Path(__file__).with_name("scientific_workflow_questions.jsonl")


def run(retrieval_mode: str = "bm25") -> dict:
    registry = SimulationRegistry(DB_PATH)
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for item in rows:
        started = time.perf_counter()
        result = run_multi_agent(item["question"], registry, LLMRouter(enabled=False), retrieval_mode, "fixed")
        statements = result.get("grounded_statements", [])
        source_types = {card["source_type"] for card in result.get("evidence_cards", [])}
        checks = {
            "route": result["task_type"] == item["task_type"],
            "review": bool(result["review"].get("approved")),
            "grounded": bool(statements) and all(statement.get("source_path") and statement.get("support") for statement in statements),
            "source_type": not item.get("source_type") or item["source_type"] in source_types,
            "answer_keyword": not item.get("must_contain") or item["must_contain"].lower() in result["answer"].lower(),
        }
        results.append({"id": item["id"], "passed": all(checks.values()), "checks": checks, "task_type": result["task_type"], "review": result["review"], "sources": sorted(source_types), "retrieval_mode_used": result.get("retrieval_mode_used"), "latency_ms": round((time.perf_counter() - started) * 1000, 3)})
    passed = sum(row["passed"] for row in results)
    latencies = [row["latency_ms"] for row in results]
    return {"questions": len(results), "passed": passed, "pass_rate": round(passed / len(results), 4), "retrieval_mode": retrieval_mode, "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 3), "p95_latency_ms": round(quantiles(latencies, n=20)[18], 3), "downgraded": sum(row["retrieval_mode_used"] != retrieval_mode for row in results if row["retrieval_mode_used"]), "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="bm25", choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
    args = parser.parse_args()
    output = ROOT / "reports" / "scientific_workflow_eval.json"
    output.write_text(json.dumps(run(args.mode), ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
