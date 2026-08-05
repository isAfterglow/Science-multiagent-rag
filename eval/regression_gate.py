"""Compare deterministic evaluation reports with the versioned release gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import ROOT


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(baseline: dict, workflow: dict, safety: dict, claims: dict | None = None, retrieval: dict | None = None) -> dict:
    agent = baseline["multi_agent"]
    safety_rule = baseline["safety"]
    checks = {
        "workflow_question_count": workflow.get("questions") == agent["questions"],
        "workflow_pass_rate": workflow.get("pass_rate", 0.0) >= agent["min_pass_rate"],
        "route_accuracy": workflow.get("route_accuracy", 0.0) >= agent["min_route_accuracy"],
        "agent_path_rate": workflow.get("agent_path_rate", 0.0) >= agent["min_agent_path_rate"],
        "reviewer_approval_rate": workflow.get("review_approval_rate", 0.0) >= agent["min_reviewer_approval_rate"],
        "safety_question_count": safety.get("questions") == safety_rule["questions"],
        "safety_pass_rate": safety.get("pass_rate", 0.0) >= safety_rule["min_pass_rate"],
    }
    if "claim_evidence" in baseline:
        claim_rule = baseline["claim_evidence"]
        checks |= {
            "claim_question_count": bool(claims) and claims.get("questions") == claim_rule["questions"],
            "claim_pass_rate": bool(claims) and claims.get("pass_rate", 0.0) >= claim_rule["min_pass_rate"],
        }
    if "scientific_retrieval" in baseline:
        retrieval_rule = baseline["scientific_retrieval"]
        metrics = (retrieval or {}).get("metrics", {})
        checks |= {
            "retrieval_question_count": metrics.get("questions") == retrieval_rule["questions"],
            "retrieval_source_recall": metrics.get("source_recall_at_5", 0.0) >= retrieval_rule["min_source_recall_at_5"],
            "retrieval_page_hit": metrics.get("page_hit_rate", 0.0) >= retrieval_rule["min_page_hit_rate"],
            "retrieval_p95_budget": metrics.get("p95_latency_ms", float("inf")) <= retrieval_rule["max_p95_latency_ms"],
        }
    p95 = workflow.get("p95_latency_ms", 0.0)
    return {
        "schema_version": "science_agent_regression_result.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "latency_warning": p95 > baseline["latency"]["p95_warning_ms"],
        "p95_latency_ms": p95,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=ROOT / "eval" / "baselines" / "deterministic-v1.json")
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--safety", type=Path, required=True)
    parser.add_argument("--claims", type=Path)
    parser.add_argument("--retrieval", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(_load(args.baseline), _load(args.workflow), _load(args.safety), _load(args.claims) if args.claims else None, _load(args.retrieval) if args.retrieval else None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
