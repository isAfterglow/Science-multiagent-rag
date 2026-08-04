from dataclasses import dataclass

import numpy as np

from app.vector_store import LocalNpyVectorStore, chunk_content_hash, corpus_fingerprint


@dataclass
class _Chunk:
    chunk_id: str
    source_type: str = "paper"
    text: str = ""
    metadata: dict | None = None

    def __post_init__(self):
        self.text = self.text or f"text for {self.chunk_id}"
        self.metadata = self.metadata or {"parent_id": "parent"}


def _chunk(chunk_id: str, source_type: str = "paper"):
    return _Chunk(chunk_id, source_type)


def test_chunk_hashes_are_content_addressed():
    left, right = _chunk("a"), _chunk("a")
    assert chunk_content_hash(left) == chunk_content_hash(right)
    assert corpus_fingerprint([left]) == corpus_fingerprint([right])
    right.text = "changed"
    assert chunk_content_hash(left) != chunk_content_hash(right)


def test_local_store_filters_and_ranks_without_vector_service():
    store = LocalNpyVectorStore("test")
    chunks = [_chunk("a", "paper"), _chunk("b", "report")]
    store.chunks = chunks
    store.vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert store.search(np.asarray([1.0, 0.0], dtype=np.float32), 2, {"paper"}) == [("a", 1.0)]
