from app.ingest import ingest
from app.llm_router import LLMRouter
from app.multi_agent import run_multi_agent
from app.registry import SimulationRegistry
from app.config import SOURCE_DIR
from app.resource_limits import acquire


def _registry(tmp_path):
    registry = SimulationRegistry(tmp_path / "registry.sqlite3")
    ingest(SOURCE_DIR, registry)
    return registry


def test_recovery_retries_once_then_refuses_unsupported_term(tmp_path):
    result = run_multi_agent("foo_unknown_scale 对 early_1_2_rmse 的影响是否明显？", _registry(tmp_path), LLMRouter(enabled=False), "bm25", "fixed")
    assert not result["review"]["approved"]
    assert len(result["recovery_actions"]) == 1
    assert result["recovery_actions"][0]["action"] == "report_priority_retry"


def test_plan_draft_is_not_execution_or_persistence(tmp_path):
    result = run_multi_agent("如何根据 early_1_2_rmse 的历史排序提出下一轮候选仿真？", _registry(tmp_path), LLMRouter(enabled=False), "bm25", "fixed")
    assert result["plan_draft"]["plan_id"].startswith("draft-")
    assert "尚未持久化或执行" in result["answer"]


def test_node_events_are_emitted_while_graph_runs(tmp_path):
    events = []
    run_multi_agent("LHS 分析中基线 early_1_2_rmse 是多少？", _registry(tmp_path), LLMRouter(enabled=False), "bm25", "fixed", events.append)
    assert [event["node"] for event in events][:2] == ["supervisor", "retriever"]
    assert events[-1]["node"] == "reviewer"


def test_busy_reranker_degrades_to_hybrid(tmp_path):
    registry = _registry(tmp_path)
    with acquire("reranker"):
        result = run_multi_agent("LHS 分析中基线 early_1_2_rmse 是多少？", registry, LLMRouter(enabled=False), "hybrid_rerank", "fixed")
    assert result["retrieval_mode_used"] == "hybrid"
