"""Evaluate the structured multi-agent graph with or without live LLM roles."""
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
from app.simulation_plan import PARAMETER_BOUNDS

QUESTIONS = Path(__file__).with_name("scientific_workflow_questions.jsonl")


def run(*, retrieval_mode: str = "dense", llm_enabled: bool = False, limit: int | None = None) -> dict:
    items = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        items = items[:limit]
    registry = SimulationRegistry(DB_PATH)
    rows: list[dict] = []
    for item in items:
        started = time.perf_counter()
        result = run_multi_agent(item["question"], registry, LLMRouter(enabled=llm_enabled), retrieval_mode)
        nodes = [event["node"] for event in result["trace"]]
        cards = result.get("evidence_cards", [])
        statements = result.get("grounded_statements", [])
        claim_verifications = result.get("claim_verifications", [])
        expected_agents = {"supervisor", "critic", "synthesizer", "semantic_critic", "reviewer"}
        if result["task_type"] in {"knowledge", "mixed"}:
            expected_agents |= {"retriever", "research_agent"}
        if result["task_type"] in {"simulation_analysis", "mixed"}:
            expected_agents.add("simulation_analyst")
        planner = result.get("planner_proposal", {})
        checks = {
            "route": result["task_type"] == item["task_type"],
            "agent_path": expected_agents.issubset(set(nodes)),
            "grounded": bool(statements) and all(row.get("source_path") and row.get("support") for row in statements),
            "claim_evidence": bool(claim_verifications) and not any(row.get("status") in {"insufficient", "conflicted"} for row in claim_verifications),
            "required_source": not item.get("source_type") or item["source_type"] in {card["source_type"] for card in cards},
            "review": bool(result.get("review", {}).get("approved")),
            "planner_whitelist": not planner or all(name in PARAMETER_BOUNDS for name in planner.get("focus_parameters", [])),
        }
        rows.append({"id": item["id"], "task_type": result["task_type"], "passed": all(checks.values()), "checks": checks, "nodes": nodes, "claim_verifications": claim_verifications, "llm_calls": result.get("llm_calls", []), "research_synthesis": result.get("research_synthesis", {}), "planner_proposal": planner, "semantic_critiques": result.get("semantic_critiques", []), "latency_ms": round((time.perf_counter() - started) * 1000, 3)})
    latencies = [row["latency_ms"] for row in rows]
    calls = [call for row in rows for call in row["llm_calls"]]
    return {
        "questions": len(rows), "llm_enabled": llm_enabled, "retrieval_mode": retrieval_mode,
        "passed": sum(row["passed"] for row in rows), "pass_rate": round(sum(row["passed"] for row in rows) / len(rows), 4) if rows else 0.0,
        "route_accuracy": round(sum(row["checks"]["route"] for row in rows) / len(rows), 4) if rows else 0.0,
        "agent_path_rate": round(sum(row["checks"]["agent_path"] for row in rows) / len(rows), 4) if rows else 0.0,
        "review_approval_rate": round(sum(row["checks"]["review"] for row in rows) / len(rows), 4) if rows else 0.0,
        "p50_latency_ms": round(sorted(latencies)[len(latencies) // 2], 3) if latencies else 0.0,
        "p95_latency_ms": round(quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0.0), 3),
        "llm_calls": len(calls), "llm_success_rate": round(sum(call["success"] for call in calls) / len(calls), 4) if calls else None,
        "results": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="dense", choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
    parser.add_argument("--llm-enabled", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "multi_agent_collaboration_eval.json")
    args = parser.parse_args()
    result = run(retrieval_mode=args.mode, llm_enabled=args.llm_enabled, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, ensure_ascii=False, indent=2))
