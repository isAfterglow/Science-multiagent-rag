from app.config import SOURCE_DIR
from app.ingest import ingest
from app.llm_router import LLMRouter
from app.multi_agent import run_multi_agent
from app.registry import SimulationRegistry
from app.trace import SCHEMA_VERSION


def _registry(tmp_path):
    registry = SimulationRegistry(tmp_path / "registry.sqlite3")
    ingest(SOURCE_DIR, registry)
    return registry


def test_trace_contract_matches_stream_events(tmp_path):
    emitted: list[dict] = []
    result = run_multi_agent(
        "cpv_front_scale 对 early_1_2_rmse 有何影响，并说明历史报告依据",
        _registry(tmp_path),
        LLMRouter(enabled=False),
        "bm25",
        "fixed",
        emitted.append,
    )

    trace = result["trace"]
    assert result["trace_id"].startswith("research-")
    assert trace == emitted
    assert result["trace_summary"]["node_count"] == len(trace)
    assert len({event["span_id"] for event in trace}) == len(trace)
    for event in trace:
        assert event["schema_version"] == SCHEMA_VERSION
        assert event["trace_id"] == result["trace_id"]
        assert event["span_id"]
        assert "parent_span_id" in event
        assert event["status"] in {"ok", "rejected"}
        assert event["elapsed_ms"] >= 0


def test_trace_records_one_bounded_recovery(tmp_path):
    result = run_multi_agent(
        "foo_unknown_scale 对 early_1_2_rmse 的影响是否明显？",
        _registry(tmp_path),
        LLMRouter(enabled=False),
        "bm25",
        "fixed",
    )
    recovery_events = [event for event in result["trace"] if event["node"] == "recovery"]
    assert len(recovery_events) == 1
    assert recovery_events[0]["status"] == "ok"
    assert result["trace_summary"]["rejected_nodes"]
