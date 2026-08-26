"""Safety regression for pre-execution validation and runtime failure taxonomy."""
from __future__ import annotations
import json
from pathlib import Path
from app.config import DB_PATH, PLAN_DB_PATH, ROOT
from app.registry import SimulationRegistry
from app.simulation_plan import PlanStore, suggest_plan, validate_plan
from app.execution import _failure_category

def run():
    registry = SimulationRegistry(DB_PATH); plan = suggest_plan(registry, n_cases=1)
    bad = plan.model_copy(deep=True); bad.cases[0].values["cpv_front_scale"] = 999
    invalid_errors = validate_plan(bad)
    rows = [
        {"id":"invalid_parameter","expected":"validation_error","actual":"validation_error" if invalid_errors else "accepted","passed":bool(invalid_errors),"errors":invalid_errors},
        {"id":"environment_failure","expected":"environment_blocked","actual":_failure_category("mpirun: operation not permitted", "", 1),"passed":_failure_category("mpirun: operation not permitted", "", 1)=="environment_blocked"},
        {"id":"runtime_failure","expected":"simulation_or_runtime_failure","actual":_failure_category("solver diverged", "", 2),"passed":_failure_category("solver diverged", "", 2)=="simulation_or_runtime_failure"},
    ]
    out={"schema_version":"execution_failure_eval.v1","cases":len(rows),"passed":sum(r["passed"] for r in rows),"pass_rate":sum(r["passed"] for r in rows)/len(rows),"results":rows}
    (ROOT/"reports"/"execution_failure_eval.json").write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    return out
if __name__ == "__main__": print(json.dumps(run(),ensure_ascii=False,indent=2))
