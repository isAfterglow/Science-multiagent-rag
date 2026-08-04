from pathlib import Path
import json

from app.analysis import parameter_correlation, top_cases
from app.config import SOURCE_DIR
from app.ingest import ingest
from app.qa import answer
from app.registry import SimulationRegistry
from app.multi_agent import run_multi_agent
from app.models import SimulationPlan
from app.simulation_plan import PlanStore, suggest_plan, template_hash, validate_plan
from app.execution import _failure_category, execute_approved_plan
from app.llm_protocol import EvidenceSummary, RoutePlan
from app.llm_router import ModelCall

def prepared_registry(tmp_path: Path) -> SimulationRegistry:
    registry = SimulationRegistry(tmp_path / "registry.sqlite3")
    ingest(SOURCE_DIR, registry)
    return registry

def test_ingestion_and_top_case(tmp_path: Path) -> None:
    registry = prepared_registry(tmp_path)
    assert len(registry.cases()) == 30
    assert top_cases(registry, "early_1_2_rmse", 1)["rows"][0]["case_id"] == "case_019"

def test_correlation_and_cited_routing(tmp_path: Path) -> None:
    registry = prepared_registry(tmp_path)
    result = parameter_correlation(registry, "early_1_2_rmse")
    assert result["correlations"]["cpv_front_scale"] < 0
    response = answer("cpv_front_scale 对 early_1_2_rmse 有何影响", registry)
    assert response["route"] == "parameter_correlation"
    assert response["citations"]

def test_retrieval_returns_source(tmp_path: Path) -> None:
    registry = prepared_registry(tmp_path)
    response = answer("LHS 分析中的 baseline early_1_2_rmse", registry, retrieval_mode="bm25")
    assert response["route"] == "hybrid_retrieval"
    assert response["citations"]

def test_multi_agent_keeps_data_and_document_evidence_separate(tmp_path: Path) -> None:
    registry = prepared_registry(tmp_path)
    response = run_multi_agent("cpv_front_scale 对 early_1_2_rmse 有何影响，并说明历史报告依据", registry, retrieval_mode="bm25")
    assert response["task_type"] == "mixed"
    assert response["evidence_cards"]
    assert response["analysis_evidence"]
    assert response["review"]["approved"]
    assert {"retriever", "simulation_analyst", "critic", "reviewer"}.issubset({event["node"] for event in response["trace"]})

def test_plan_validation_and_approval_gate(tmp_path: Path, monkeypatch) -> None:
    registry = prepared_registry(tmp_path)
    plan = suggest_plan(registry, n_cases=1)
    assert plan.template_sha256 == template_hash()
    assert not validate_plan(plan)
    invalid = plan.model_copy(deep=True); invalid.cases[0].values["cpv_front_scale"] = 9.0
    assert validate_plan(invalid)
    store = PlanStore(tmp_path / "plans.sqlite3")
    store.create(plan)
    try:
        execute_approved_plan(plan.plan_id, store, registry, dry_run=True)
        assert False, "unapproved plan must not execute"
    except PermissionError:
        pass
    store.decide(plan.plan_id, "approved", "reviewer")
    import app.execution as execution
    monkeypatch.setattr(execution, "RUNS_DIR", tmp_path / "runs")
    result = execute_approved_plan(plan.plan_id, store, registry, dry_run=True)
    assert result["preview_cases"] == 1
    assert (tmp_path / "runs" / plan.plan_id / "candidate-1" / "case1_fiat_walltemp_nominal.i").exists()
    manifest = json.loads((tmp_path / "runs" / plan.plan_id / "candidate-1" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mpi_processes"] == 4

def test_mpi_permission_failure_is_not_misclassified_as_model_failure() -> None:
    assert _failure_category("", "HYDU_sock_listen: cannot open socket (Operation not permitted)", 255) == "environment_blocked"

def test_llm_assistance_is_constrained_to_validated_protocol(tmp_path: Path) -> None:
    class FakeRouter:
        enabled = True
        telemetry = [ModelCall("planner", "fake", 1.0, True)]
        def call_json(self, role, _system, _user, schema):
            if role == "router": return RoutePlan(task_type="mixed", needs_registry_analysis=True, required_sources=["report"], needs_experiment=False, retrieval_query="LHS historical report cpv_front_scale", analysis_metric="early_1_2_rmse", reason="test")
            return EvidenceSummary(summary="已根据提供摘录整理。", limitations=["不能证明因果关系。"])
    registry = prepared_registry(tmp_path)
    result = run_multi_agent("cpv_front_scale 对 early_1_2_rmse 有何影响，并说明历史报告依据", registry, router=FakeRouter(), retrieval_mode="bm25")
    assert result["llm_calls"]
    assert result["llm_evidence_summary"]["summary"] == "已根据提供摘录整理。"
    assert "已根据提供摘录整理。" not in result["answer"]
    assert result["review"]["approved"]
