from app.retrieval import resolve_service_tier
from eval.retrieval_routing_eval import lexical_confidence


def test_router_rejects_generic_table_word_and_accepts_exact_technical_query():
    assert not lexical_confidence("请解释这个表格的机理")[0]
    assert lexical_confidence("FIATC table Variables Equation Number 30 kW")[0]


def test_service_tiers_have_explicit_cost_quality_mapping():
    assert resolve_service_tier("fast", "dense") == "bm25"
    assert resolve_service_tier("default", "dense") == "dense"
    assert resolve_service_tier("precision", "dense") == "source_fusion"
