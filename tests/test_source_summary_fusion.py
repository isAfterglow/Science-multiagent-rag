from __future__ import annotations

from app.retrieval import Chunk, HybridRetriever


def test_source_fusion_keeps_full_page_candidates(monkeypatch):
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.chunks = [
        Chunk("a", "paper", "a.pdf", "first", 1, 1, {"source_id": "a", "title": "Alpha"}),
        Chunk("b", "paper", "b.pdf", "second", 1, 1, {"source_id": "b", "title": "Beta"}),
    ]
    monkeypatch.setattr(retriever, "_source_summary_ranking", lambda _query, _sources: {"b": 1, "a": 2})
    ranking, details = retriever._source_fusion("question", [0, 1])
    assert set(ranking) == {0, 1}
    assert details[1]["source_summary_rank"] == 1
