import re

from app.config import SOURCE_DIR
from app.ingest import ingest
from app.llm_protocol import PlannerProposal, ResearchClaim, ResearchSynthesis, RoutePlan, SemanticCritique, SemanticCritiqueItem
from app.llm_router import ModelCall
from app.multi_agent import _validated_planner, _validated_research, _validated_semantic_critique, run_multi_agent
from app.registry import SimulationRegistry


def _registry(tmp_path):
    registry = SimulationRegistry(tmp_path / "registry.sqlite3")
    ingest(SOURCE_DIR, registry)
    registry.add_document("test:fiatc", "paper", "fiatc.pdf", "FIATC equation table maps Variables to Equation Number on page 4.", {"source_id": "test_fiatc", "page": 4, "authority": "C"})
    return registry


class _CollaboratingRouter:
    enabled = True
    telemetry = [ModelCall("router", "fake", 1.0, True)]

    def call_json(self, role, _system, user, _schema):
        if role == "router":
            return RoutePlan(task_type="knowledge", needs_registry_analysis=False, required_sources=["paper"], needs_experiment=False, retrieval_query="FIATC table", analysis_metric="early_1_2_rmse", reason="test")
        evidence_index = re.search(r"EVIDENCE_INDEX: (\d+)", user)
        if role == "research":
            return ResearchSynthesis(claims=[ResearchClaim(text="证据归纳。", evidence_indexes=[int(evidence_index.group(1))], limitations=[])])
        if role == "critic":
            return SemanticCritique(issues=[SemanticCritiqueItem(issue_type="causality", severity="warning", message="不能由单一摘录推出因果。", evidence_indexes=[])])
        return None


def test_llm_research_and_semantic_critic_are_traceable(tmp_path):
    result = run_multi_agent("FIATC 论文中的方程表在哪一页？", _registry(tmp_path), router=_CollaboratingRouter(), retrieval_mode="bm25")
    nodes = {event["node"] for event in result["trace"]}
    assert {"research_agent", "semantic_critic", "reviewer"}.issubset(nodes)
    assert result["research_synthesis"]["claims"]
    semantic_events = [event for event in result["trace"] if event["node"] == "semantic_critic"]
    assert any(event["accepted"] and event["issue_count"] == 1 for event in semantic_events)


def test_llm_agent_protocols_reject_unknown_evidence_and_parameters():
    cards = [{"retrieval": {"chunk_id": "known"}}]
    assert _validated_research(ResearchSynthesis(claims=[ResearchClaim(text="x", evidence_indexes=[9])]), cards) is None
    assert _validated_semantic_critique(SemanticCritique(issues=[SemanticCritiqueItem(issue_type="overreach", severity="error", message="x", evidence_indexes=[9])]), cards) is None
    assert _validated_planner(PlannerProposal(target_metric="early_1_2_rmse", focus_parameters=["unknown"], rationale="仅供测试的受限规划建议。", requires_human_approval=True), "early_1_2_rmse", True) is None
