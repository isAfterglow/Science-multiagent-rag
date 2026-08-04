from app.config import SOURCE_DIR
from app.experiment_cycle import design_experiment, validate_and_report_result
from app.ingest import ingest
from app.mcp_server import TOOLS, dispatch_tool
from app.registry import SimulationRegistry
from app.simulation_plan import PlanStore


def _registry(tmp_path):
    registry = SimulationRegistry(tmp_path / "registry.sqlite3")
    ingest(SOURCE_DIR, registry)
    return registry


def test_gap_draft_confirmation_and_preview_report(tmp_path, monkeypatch):
    registry = _registry(tmp_path); designed = design_experiment("新工况下 cpv_front_scale 对 early_1_2_rmse 的影响没有历史记录，需要验证。", registry)
    assert designed["gap"]["status"] == "needs_experiment"
    store = PlanStore(tmp_path / "plans.sqlite3"); pending = store.create(__import__("app.models", fromlist=["SimulationPlan"]).SimulationPlan.model_validate(designed["plan"]))
    assert pending["status"] == "pending"
    approved = store.decide(pending["plan_id"], "approved", "test")
    import app.execution as execution
    monkeypatch.setattr(execution, "RUNS_DIR", tmp_path / "runs")
    execution.execute_approved_plan(approved["plan_id"], store, registry, dry_run=True)
    report = validate_and_report_result(approved["plan_id"], store, registry)
    assert "dry-run" in report["report"]


def test_mcp_exposes_typed_tools_without_sql_or_real_execution(tmp_path, monkeypatch):
    names = {tool.name for tool in TOOLS}
    assert {"search_evidence", "plot_parameter_scatter", "create_experiment_draft", "execute_plan_preview"}.issubset(names)
    assert "execute_sql" not in names and "execute_plan_real" not in names
    import app.plotting as plotting
    import app.mcp_server as mcp_server
    registry = _registry(tmp_path)
    monkeypatch.setattr(plotting, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(mcp_server, "DB_PATH", registry.path)
    # MCP uses the configured registry for read-only analysis and returns a
    # traceable artifact rather than executable code.
    result = dispatch_tool("plot_metric_ranking_bar", {"metric": "early_1_2_rmse", "top_k": 3})
    assert result["chart_type"] == "metric_ranking_bar"
    assert (tmp_path / "artifacts" / f"{result['artifact_id']}.png").exists()
