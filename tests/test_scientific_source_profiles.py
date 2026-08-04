from app.multi_agent import _evidence_requirements
from app.retrieval import Chunk, HybridRetriever


def test_specific_english_profile_token_beats_generic_repeated_token():
    query = "请检索 Advanced TUFROC 的双层结构"
    generic = Chunk("generic", "paper", "generic.pdf", "", 1, 1, {"title": "TUFROC Thermal Protection System", "topics": ["TUFROC"], "aliases": ["TUFROC"]})
    advanced = Chunk("advanced", "paper", "advanced.pdf", "", 1, 1, {"title": "Advanced TUFROC Thermal Protection System", "topics": ["TUFROC"], "aliases": ["先进TUFROC"]})
    assert HybridRetriever._profile_score(query, advanced) > HybridRetriever._profile_score(query, generic)


def test_explicit_scan_request_does_not_add_generic_paper_requirement():
    requirements = _evidence_requirements("扫描版多层壁热防护系统资料中，详细说明从哪一页开始？", analysis=False)
    assert [item.kind for item in requirements] == ["scan_report"]


def test_english_experiment_condition_routes_to_public_paper():
    requirements = _evidence_requirements("TPS arc jet shear testing 的数值表", analysis=False)
    assert [item.kind for item in requirements] == ["paper"]
