"""Replaceable dense-vector backends with content-addressed Milvus sync."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import Any, Protocol

import numpy as np

from app.config import MILVUS_COLLECTION, MILVUS_INDEX_TYPE, MILVUS_TOKEN, MILVUS_URI, ROOT, VECTOR_BACKEND


class VectorStore(Protocol):
    backend: str

    def sync(self, chunks: list[Any], vectors: np.ndarray) -> dict[str, Any]: ...
    def search(self, vector: np.ndarray, limit: int, source_types: set[str] | None = None) -> list[tuple[str, float]]: ...
    def status(self) -> dict[str, Any]: ...


def corpus_fingerprint(chunks: list[Any]) -> str:
    return hashlib.sha256(json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


def chunk_content_hash(chunk: Any) -> str:
    payload = {"id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class LocalNpyVectorStore:
    backend = "local"

    def __init__(self, strategy: str) -> None:
        self.strategy = strategy
        self.chunks: list[Any] = []
        self.vectors: np.ndarray | None = None
        self._fingerprint = ""

    def sync(self, chunks: list[Any], vectors: np.ndarray) -> dict[str, Any]:
        fingerprint = corpus_fingerprint(chunks)
        root = ROOT / "data" / "index"
        root.mkdir(parents=True, exist_ok=True)
        vector_path = root / f"bge_m3_{self.strategy}_{fingerprint}.npy"
        metadata_path = root / f"bge_m3_{self.strategy}_{fingerprint}.json"
        if vector_path.exists() and metadata_path.exists():
            vectors = np.load(vector_path)
            cache_hit = True
        else:
            np.save(vector_path, vectors)
            metadata_path.write_text(json.dumps({"chunk_strategy": self.strategy, "count": len(chunks), "fingerprint": fingerprint}, ensure_ascii=False), encoding="utf-8")
            cache_hit = False
        self.chunks, self.vectors, self._fingerprint = chunks, vectors, fingerprint
        return {"backend": self.backend, "status": "ready", "count": len(chunks), "fingerprint": fingerprint, "cache_hit": cache_hit}

    def search(self, vector: np.ndarray, limit: int, source_types: set[str] | None = None) -> list[tuple[str, float]]:
        if self.vectors is None:
            return []
        allowed = [(index, chunk) for index, chunk in enumerate(self.chunks) if not source_types or chunk.source_type in source_types]
        ranking = sorted(((chunk.chunk_id, float(self.vectors[index] @ vector)) for index, chunk in allowed), key=lambda item: item[1], reverse=True)
        return ranking[:limit]

    def status(self) -> dict[str, Any]:
        return {"backend": self.backend, "status": "ready" if self.vectors is not None else "empty", "count": len(self.chunks), "fingerprint": self._fingerprint}


class MilvusVectorStore:
    """Milvus backend. SQLite remains the evidence source of truth.

    Only chunk IDs, vectors, filters and version hashes are copied to Milvus.
    Full excerpts, bboxes and table CSV are fetched from the in-process chunk
    map after ANN recall, avoiding duplicated scientific evidence payloads.
    """

    backend = "milvus"

    def __init__(self, *, uri: str = MILVUS_URI, collection: str = MILVUS_COLLECTION, token: str = MILVUS_TOKEN, index_type: str = MILVUS_INDEX_TYPE) -> None:
        self.uri, self.collection, self.token, self.index_type = uri, collection, token, index_type
        self.client: Any | None = None
        self.dimension: int | None = None
        self._status: dict[str, Any] = {"backend": self.backend, "status": "not_connected", "uri": uri, "collection": collection, "count": 0}

    def _client(self) -> Any:
        if self.client is not None:
            return self.client
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise RuntimeError("pymilvus is not installed; install requirements.txt before enabling VECTOR_BACKEND=milvus") from exc
        self._data_type = DataType
        kwargs: dict[str, Any] = {"uri": self.uri}
        if self.token:
            kwargs["token"] = self.token
        self.client = MilvusClient(**kwargs)
        return self.client

    def _ensure_collection(self, dimension: int) -> None:
        client = self._client()
        if client.has_collection(self.collection):
            client.load_collection(self.collection)
            self.dimension = dimension
            return
        schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", self._data_type.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("vector", self._data_type.FLOAT_VECTOR, dim=dimension)
        schema.add_field("source_type", self._data_type.VARCHAR, max_length=64)
        schema.add_field("document_id", self._data_type.VARCHAR, max_length=512)
        schema.add_field("content_hash", self._data_type.VARCHAR, max_length=64)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type=self.index_type, metric_type="IP")
        client.create_collection(collection_name=self.collection, schema=schema, index_params=index_params)
        client.load_collection(self.collection)
        self.dimension = dimension

    def sync(self, chunks: list[Any], vectors: np.ndarray) -> dict[str, Any]:
        if len(chunks) != len(vectors):
            raise ValueError("chunk/vector count mismatch")
        started = time.perf_counter()
        self._ensure_collection(int(vectors.shape[1]))
        client = self._client()
        # Reading the lightweight fields allows true content-hash upsert: an
        # unchanged corpus is not re-embedded or resent to the vector service.
        existing = {row["chunk_id"]: row.get("content_hash", "") for row in client.query(self.collection, filter="", output_fields=["chunk_id", "content_hash"], limit=16384)}
        current = {chunk.chunk_id: chunk_content_hash(chunk) for chunk in chunks}
        stale = sorted(set(existing) - set(current))
        if stale:
            quoted = ", ".join(json.dumps(item) for item in stale)
            client.delete(self.collection, filter=f"chunk_id in [{quoted}]")
        changed = [index for index, chunk in enumerate(chunks) if existing.get(chunk.chunk_id) != current[chunk.chunk_id]]
        if changed:
            payload = [{"chunk_id": chunks[index].chunk_id, "vector": vectors[index].astype(float).tolist(), "source_type": chunks[index].source_type, "document_id": str(chunks[index].metadata.get("parent_id") or chunks[index].chunk_id.rsplit(":parent_child:", 1)[0]), "content_hash": current[chunks[index].chunk_id]} for index in changed]
            client.upsert(self.collection, data=payload)
        count = int(client.get_collection_stats(self.collection).get("row_count", len(current)))
        self._status = {"backend": self.backend, "status": "ready", "uri": self.uri, "collection": self.collection, "index_type": self.index_type, "count": count, "dimension": int(vectors.shape[1]), "fingerprint": corpus_fingerprint(chunks), "upserted": len(changed), "deleted": len(stale), "sync_latency_ms": round((time.perf_counter() - started) * 1000, 3)}
        return self._status

    def search(self, vector: np.ndarray, limit: int, source_types: set[str] | None = None) -> list[tuple[str, float]]:
        client = self._client()
        filters = ""
        if source_types:
            filters = "source_type in [" + ", ".join(json.dumps(value) for value in sorted(source_types)) + "]"
        result = client.search(collection_name=self.collection, data=[vector.astype(float).tolist()], anns_field="vector", limit=limit, filter=filters, output_fields=["chunk_id", "source_type"])
        hits = result[0] if result else []
        return [(str(hit["entity"]["chunk_id"]), float(hit["distance"])) for hit in hits]

    def status(self) -> dict[str, Any]:
        try:
            client = self._client()
            if client.has_collection(self.collection):
                self._status["count"] = int(client.get_collection_stats(self.collection).get("row_count", 0))
                self._status["status"] = "ready"
            else:
                self._status["status"] = "collection_missing"
        except Exception as exc:
            self._status.update({"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"})
        return dict(self._status)


def configured_vector_status() -> dict[str, Any]:
    """Return a cheap, truthful backend health snapshot for the workbench."""
    if VECTOR_BACKEND == "milvus":
        return MilvusVectorStore().status()
    return {"backend": "local", "status": "configured", "index_root": str(ROOT / "data" / "index")}
