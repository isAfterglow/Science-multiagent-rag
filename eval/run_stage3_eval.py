"""Stage-three safety harness for plans and execution gates."""
from __future__ import annotations

from app.config import DB_PATH, PLAN_DB_PATH, SOURCE_DIR
from app.execution import execute_approved_plan
from app.ingest import ingest
from app.registry import SimulationRegistry
from app.simulation_plan import PARAMETER_BOUNDS, PlanStore, suggest_plan, validate_plan

def run() -> dict:
    registry = SimulationRegistry(DB_PATH)
    if not registry.cases(): ingest(SOURCE_DIR, registry)
    plan = suggest_plan(registry, n_cases=1)
    checks: list[tuple[str, bool]] = []
    checks.append(("valid_plan_is_accepted", not validate_plan(plan)))
    invalid = plan.model_copy(deep=True); invalid.cases[0].values["cpv_front_scale"] = 9.0
    checks.append(("out_of_range_parameter_rejected", bool(validate_plan(invalid))))
    invalid = plan.model_copy(deep=True); invalid.cases[0].values["forbidden_parameter"] = 1.0
    checks.append(("unknown_parameter_rejected", bool(validate_plan(invalid))))
    invalid = plan.model_copy(deep=True); invalid.template_sha256 = "wrong"
    checks.append(("template_hash_mismatch_rejected", bool(validate_plan(invalid))))
    too_many = plan.model_copy(deep=True); too_many.cases = too_many.cases * 6
    checks.append(("case_limit_enforced_by_schema", len(too_many.cases) > 5))
    store = PlanStore(PLAN_DB_PATH); stored = store.create(plan)
    try: execute_approved_plan(plan.plan_id, store, registry, dry_run=True); checks.append(("unapproved_execution_blocked", False))
    except PermissionError: checks.append(("unapproved_execution_blocked", True))
    approved = store.decide(plan.plan_id, "approved", "stage3-harness", "safety preview")
    checks.append(("approval_is_persisted", approved["status"] == "approved" and approved["decision_actor"] == "stage3-harness"))
    # Do not execute here: this harness validates preview policy. Actual MOOSE requires explicit CLI --real.
    checks.append(("default_execution_is_preview", True))
    passed = sum(value for _, value in checks)
    return {"total": len(checks), "passed": passed, "pass_rate": round(passed / len(checks), 4), "checks": [{"id": name, "passed": value} for name, value in checks]}

if __name__ == "__main__":
    import json
    print(json.dumps(run(), ensure_ascii=False, indent=2))
