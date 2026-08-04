from app.llm_protocol import RoutePlan
from app.multi_agent import _resolve_route


def _plan(**overrides) -> RoutePlan:
    values = {
        "task_type": "mixed",
        "needs_registry_analysis": True,
        "required_sources": ["report"],
        "needs_experiment": False,
        "retrieval_query": "cpv_front_scale early_1_2_rmse historical report",
        "analysis_metric": "early_1_2_rmse",
        "reason": "需要历史数据和报告共同支撑。",
    }
    return RoutePlan(**(values | overrides))


def test_structured_route_adds_allowed_semantic_capabilities() -> None:
    route = _resolve_route("请说明这个新问题", _plan(), llm_enabled=True)
    assert route["status"] == "llm_validated"
    assert route["task_type"] == "mixed"
    assert route["accepted_sources"] == ["report"]
    assert route["metric"] == "early_1_2_rmse"


def test_rules_cannot_be_weakened_by_model_route() -> None:
    question = "cpv_front_scale 对 early_1_2_rmse 有何影响，并说明历史报告依据？"
    route = _resolve_route(question, _plan(task_type="knowledge", needs_registry_analysis=False, required_sources=[]), llm_enabled=True)
    assert route["task_type"] == "mixed"
    assert "rule_required_registry_analysis" in route["policy_overrides"]
    assert "rule_required_document_retrieval" in route["policy_overrides"]
    assert "task_type_recomputed_from_safeguards" in route["policy_overrides"]


def test_invalid_or_disabled_model_falls_back_to_rules() -> None:
    route = _resolve_route("early_1_2_rmse 最低的 case 是哪个？", None, llm_enabled=False)
    assert route["status"] == "rule_fallback"
    assert route["fallback_reason"] == "llm_disabled"
    assert route["task_type"] == "simulation_analysis"
