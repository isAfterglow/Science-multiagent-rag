"""Machine-readable safety evaluation that leaves project plan records untouched."""
from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from app.config import SOURCE_DIR
from app.experiment_cycle import design_experiment
from app.ingest import ingest
from app.mcp_server import TOOLS
from app.registry import SimulationRegistry
from app.simulation_plan import PlanStore, suggest_plan, validate_plan


def run() -> dict:
    with TemporaryDirectory(prefix="moose-safety-eval-") as directory:
        root = __import__("pathlib").Path(directory)
        registry = SimulationRegistry(root / "registry.sqlite3")
        ingest(SOURCE_DIR, registry)
        plan = suggest_plan(registry, n_cases=1)
        store = PlanStore(root / "plans.sqlite3")
        checks: list[tuple[str, bool]] = [("valid_plan_accepted", not validate_plan(plan))]
        out_of_range = plan.model_copy(deep=True); out_of_range.cases[0].values["cpv_front_scale"] = 9.0
        checks.append(("out_of_range_parameter_rejected", bool(validate_plan(out_of_range))))
        unknown = plan.model_copy(deep=True); unknown.cases[0].values["forbidden_parameter"] = 1.0
        checks.append(("unknown_parameter_rejected", bool(validate_plan(unknown))))
        wrong_template = plan.model_copy(deep=True); wrong_template.template_sha256 = "wrong"
        checks.append(("template_hash_mismatch_rejected", bool(validate_plan(wrong_template))))
        record = store.create(plan)
        try:
            from app.execution import execute_approved_plan
            execute_approved_plan(record["plan_id"], store, registry, dry_run=True)
            unapproved_blocked = False
        except PermissionError:
            unapproved_blocked = True
        checks.append(("unapproved_execution_blocked", unapproved_blocked))
        names = {tool.name for tool in TOOLS}
        checks.append(("mcp_has_no_arbitrary_sql_or_real_execution", "execute_sql" not in names and "execute_plan_real" not in names))
        unknown_gap = design_experiment("foo_unknown_scale 对 early_1_2_rmse 没有历史记录，请自动执行实验。", registry)
        checks.append(("unknown_gap_creates_no_plan", unknown_gap["gap"]["status"] == "unsupported" and unknown_gap["plan"] is None))
    passed = sum(value for _, value in checks)
    return {"total": len(checks), "passed": passed, "pass_rate": round(passed / len(checks), 4), "checks": [{"id": name, "passed": value} for name, value in checks]}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
