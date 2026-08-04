from __future__ import annotations

import json
from pathlib import Path

from app.qa import answer
from app.registry import SimulationRegistry

QUESTIONS = Path(__file__).with_name("questions.jsonl")

def run(registry: SimulationRegistry) -> dict:
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    outcomes = []
    for item in rows:
        result = answer(item["question"], registry)
        passed = bool(result["citations"])
        if item["type"] == "top_case": passed = passed and result["route"] == "top_cases" and bool(result["answer"]["rows"])
        elif item["type"] == "correlation": passed = passed and result["route"] == "parameter_correlation" and bool(result["answer"]["correlations"])
        elif item["type"] == "case_detail": passed = passed and result["answer"]["case_id"] == item["case_id"]
        elif item["type"] == "retrieval": passed = passed and result["route"] == "hybrid_retrieval"
        outcomes.append({"id": item["id"], "type": item["type"], "passed": passed, "route": result["route"]})
    passed = sum(item["passed"] for item in outcomes)
    return {"total": len(outcomes), "passed": passed, "pass_rate": round(passed / len(outcomes), 4), "by_type": {kind: {"total": sum(item["type"] == kind for item in outcomes), "passed": sum(item["type"] == kind and item["passed"] for item in outcomes)} for kind in sorted({item["type"] for item in outcomes})}, "results": outcomes}
