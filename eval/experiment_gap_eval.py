from __future__ import annotations

import json
from pathlib import Path

from app.config import DB_PATH, ROOT
from app.experiment_cycle import design_experiment
from app.registry import SimulationRegistry


def run() -> dict:
    registry = SimulationRegistry(DB_PATH); path = Path(__file__).with_name("experiment_gap_questions.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for row in rows:
        result = design_experiment(row["question"], registry, row.get("metric", "early_1_2_rmse"))
        passed = result["gap"]["status"] == row["expected"]
        if row["expected"] == "needs_experiment": passed = passed and bool(result["plan"]) and result["plan"]["plan_id"].startswith("draft-")
        results.append({"id": row["id"], "passed": passed, "status": result["gap"]["status"]})
    passed = sum(row["passed"] for row in results)
    output = {"questions": len(results), "passed": passed, "pass_rate": round(passed / len(results), 4), "results": results}
    (ROOT / "reports" / "experiment_gap_eval.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
