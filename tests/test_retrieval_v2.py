from app.config import SOURCE_DIR
from app.ingest import ingest
from app.registry import SimulationRegistry
from app.retrieval import HybridRetriever, build_chunks


def _registry(tmp_path):
    registry = SimulationRegistry(tmp_path / "registry.sqlite3")
    ingest(SOURCE_DIR, registry)
    return registry


def test_chunk_strategies_preserve_traceable_line_ranges(tmp_path):
    registry = _registry(tmp_path)
    documents = registry.documents()
    for strategy in ("document", "fixed", "structure"):
        chunks = build_chunks(documents, strategy)
        assert chunks
        assert all(chunk.start_line >= 1 and chunk.end_line >= chunk.start_line for chunk in chunks)


def test_bm25_evidence_has_retrieval_provenance(tmp_path):
    registry = _registry(tmp_path)
    cards = HybridRetriever(registry, mode="bm25", chunk_strategy="fixed").search("LHS 分析中基线 early_1_2_rmse 是多少")
    assert cards
    assert all(card.retrieval["mode"] == "bm25" for card in cards)
    assert all(card.retrieval["chunk_id"] for card in cards)
    assert len({card.source_path for card in cards}) == len(cards)
