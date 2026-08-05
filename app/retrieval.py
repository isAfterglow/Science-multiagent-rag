"""Local BGE-M3 + BM25 + RRF retrieval with chunking and reranking experiments."""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Literal

import numpy as np
from rank_bm25 import BM25Okapi

from app.config import CHUNK_STRATEGY, EMBEDDING_DEVICE, EMBEDDING_MODEL_PATH, RETRIEVAL_MODE, RETRIEVAL_TIER, RERANKER_DEVICE, RERANKER_MODEL_PATH, ROOT, VECTOR_BACKEND
from app.models import EvidenceCard
from app.resource_limits import acquire, acquire_inference
from app.registry import SimulationRegistry
from app.vector_store import LocalNpyVectorStore, MilvusVectorStore, VectorStore, chunk_content_hash, corpus_fingerprint

RetrievalMode = Literal["bm25", "dense", "dense_page", "hybrid", "hybrid_rerank", "source_fusion"]
ChunkStrategy = Literal["document", "fixed", "structure", "parent_child"]
_EMBEDDER = None
_RERANKER = None
_TOKENIZER = None
_DEVICE_STATUS: dict[str, str] = {}
_MODEL_INIT_LOCK = Lock()


def resolve_service_tier(tier: str, configured_mode: str) -> str:
    """Map explicit latency/quality tiers without hiding the selected mode."""
    return {"fast": "bm25", "default": configured_mode, "precision": "source_fusion"}.get(tier, configured_mode)

def _resolve_device(requested: str, label: str) -> str:
    if requested == "cpu": return "cpu"
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
    except Exception: pass
    _DEVICE_STATUS[label] = "cpu_fallback: CUDA unavailable"
    return "cpu"

def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[\u4e00-\u9fff]+|\d+(?:\.\d+)?", text.lower())

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_type: str
    source_path: str
    text: str
    start_line: int
    end_line: int
    metadata: dict[str, Any] = field(default_factory=dict)

def _fixed_chunks(document: dict[str, str], size: int = 850, overlap: int = 140) -> list[Chunk]:
    text = document["content"]
    chunks = []
    for start in range(0, len(text), size - overlap):
        part = text[start:start + size]
        if not part.strip(): continue
        start_line = text[:start].count("\n") + 1
        chunks.append(Chunk(f"{document['document_id']}:fixed:{len(chunks)}", document["source_type"], document["source_path"], part, start_line, start_line + part.count("\n"), document.get("metadata", {})))
        if start + size >= len(text): break
    return chunks

def _structure_chunks(document: dict[str, str]) -> list[Chunk]:
    text = document["content"]
    if document["source_type"] == "input_deck":
        pieces = list(re.finditer(r"(?m)^\s*\[[^\n]+\]", text))
        spans = [(match.start(), pieces[i + 1].start() if i + 1 < len(pieces) else len(text)) for i, match in enumerate(pieces)] or [(0, len(text))]
    elif document["source_type"] in {"run_log", "run_status", "script"}:
        lines = text.splitlines(keepends=True); spans = []
        for start in range(0, len(lines), 28):
            prefix = "".join(lines[:start]); part = "".join(lines[start:start + 32]); spans.append((len(prefix), len(prefix) + len(part)))
    else:
        paragraphs = list(re.finditer(r"(?s).{1,1200}(?:\n\s*\n|$)", text))
        spans = [(match.start(), match.end()) for match in paragraphs] or [(0, len(text))]
    chunks = []
    for index, (start, end) in enumerate(spans):
        part = text[start:end]
        if part.strip():
            line = text[:start].count("\n") + 1
            chunks.append(Chunk(f"{document['document_id']}:structure:{index}", document["source_type"], document["source_path"], part, line, line + part.count("\n"), document.get("metadata", {})))
    return chunks


def _parent_child_chunks(document: dict[str, Any], max_tokens: int = 384, overlap_tokens: int = 64) -> list[Chunk]:
    """Create BGE-token-aware children while retaining page/section parents."""
    text = document["content"]
    metadata = dict(document.get("metadata", {}))
    tokenizer = _tokenizer()
    segments = [part.strip() for part in re.split(r"(?:\n\s*\n|(?<=[。！？.!?])\s+)", text) if part.strip()]
    if not segments:
        return []
    chunks: list[Chunk] = []
    current: list[int] = []
    current_start = 0

    def emit(ids: list[int], start_offset: int) -> None:
        # Standalone one-word headings and layout fragments dilute both BM25
        # and dense vectors; meaningful tables/paragraphs comfortably exceed it.
        if len(ids) < 8:
            return
        part = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        if not part:
            return
        child_metadata = {**metadata, "child_index": len(chunks), "token_count": len(ids), "parent_id": metadata.get("parent_id") or document["document_id"]}
        line = text[:start_offset].count("\n") + 1
        chunks.append(Chunk(f"{document['document_id']}:parent_child:{len(chunks)}", document["source_type"], document["source_path"], part, line, line + part.count("\n"), child_metadata))

    cursor = 0
    for segment in segments:
        segment_ids = tokenizer.encode(segment, add_special_tokens=False)
        segment_start = text.find(segment, cursor)
        cursor = segment_start + len(segment) if segment_start >= 0 else cursor
        while segment_ids:
            remaining = max_tokens - len(current)
            if remaining == 0:
                emit(current, current_start)
                current = current[-overlap_tokens:]
                current_start = max(0, segment_start)
                remaining = max_tokens - len(current)
            if not current:
                current_start = max(0, segment_start)
            current.extend(segment_ids[:remaining])
            segment_ids = segment_ids[remaining:]
            if len(current) == max_tokens:
                emit(current, current_start)
                current = current[-overlap_tokens:]
                current_start = max(0, segment_start)
    emit(current, current_start)
    return chunks

def build_chunks(documents: list[dict[str, Any]], strategy: ChunkStrategy) -> list[Chunk]:
    output: list[Chunk] = []
    for document in documents:
        if strategy == "document": output.append(Chunk(f"{document['document_id']}:document:0", document["source_type"], document["source_path"], document["content"], 1, document["content"].count("\n") + 1, document.get("metadata", {})))
        elif strategy == "fixed": output.extend(_fixed_chunks(document))
        elif strategy == "structure": output.extend(_structure_chunks(document))
        else: output.extend(_parent_child_chunks(document))
    return output

def _embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        with _MODEL_INIT_LOCK:
            if _EMBEDDER is None:
                if not EMBEDDING_MODEL_PATH.exists(): raise FileNotFoundError(f"Embedding model missing: {EMBEDDING_MODEL_PATH}")
                from FlagEmbedding import BGEM3FlagModel
                # Index construction is an offline workload. A larger CPU batch avoids
                # turning a small local corpus into hundreds of transformer calls.
                device = _resolve_device(EMBEDDING_DEVICE, "embedding")
                try:
                    _EMBEDDER = BGEM3FlagModel(str(EMBEDDING_MODEL_PATH), use_fp16=device == "cuda", devices=device, batch_size=16 if device == "cuda" else 32, query_max_length=512, passage_max_length=512)
                    _DEVICE_STATUS["embedding"] = device
                except Exception as exc:
                    if device != "cuda": raise
                    _DEVICE_STATUS["embedding"] = f"cpu_fallback: {type(exc).__name__}"
                    _EMBEDDER = BGEM3FlagModel(str(EMBEDDING_MODEL_PATH), use_fp16=False, devices="cpu", batch_size=32, query_max_length=512, passage_max_length=512)
    return _EMBEDDER

def _reranker():
    global _RERANKER
    if _RERANKER is None:
        with _MODEL_INIT_LOCK:
            if _RERANKER is None:
                if not RERANKER_MODEL_PATH.exists(): raise FileNotFoundError(f"Reranker model missing: {RERANKER_MODEL_PATH}")
                from FlagEmbedding import FlagReranker
                device = _resolve_device(RERANKER_DEVICE, "reranker")
                try:
                    _RERANKER = FlagReranker(str(RERANKER_MODEL_PATH), use_fp16=device == "cuda", devices=device, batch_size=8 if device == "cuda" else 16, max_length=512, normalize=True)
                    _DEVICE_STATUS["reranker"] = device
                except Exception as exc:
                    if device != "cuda": raise
                    _DEVICE_STATUS["reranker"] = f"cpu_fallback: {type(exc).__name__}"
                    _RERANKER = FlagReranker(str(RERANKER_MODEL_PATH), use_fp16=False, devices="cpu", batch_size=16, max_length=512, normalize=True)
    return _RERANKER


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        with _MODEL_INIT_LOCK:
            if _TOKENIZER is None:
                from transformers import AutoTokenizer
                _TOKENIZER = AutoTokenizer.from_pretrained(str(EMBEDDING_MODEL_PATH), local_files_only=True)
    return _TOKENIZER


def _rerank_scores(query: str, passages: list[str]) -> list[float]:
    """Score query-passage pairs without FlagEmbedding's removed tokenizer API.

    FlagEmbedding 1.3 calls ``prepare_for_model``, which was removed by the
    Transformers version installed in this environment. Its public model and
    tokenizer are stable, so batching them directly preserves BGE reranking
    semantics while avoiding a version pin in the application runtime.
    """
    import torch

    # CPU reranking is a final precision stage, not a second retrieval pass.
    # Restrict threads and sequence length so it remains usable interactively.
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    reranker = _reranker()
    tokenizer, model = reranker.tokenizer, reranker.model
    device = _DEVICE_STATUS.get("reranker", "cpu")
    device = "cuda" if device == "cuda" else "cpu"
    model.to(device).eval()
    scores: list[float] = []
    for start in range(0, len(passages), 16):
        batch = passages[start:start + 16]
        inputs = tokenizer([query] * len(batch), batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs, return_dict=True).logits.view(-1).float()
        scores.extend(torch.sigmoid(logits).tolist())
    return scores

class HybridRetriever:
    """Retrieves from SQLite-derived chunks through a replaceable vector backend."""
    def __init__(self, registry: SimulationRegistry, mode: RetrievalMode | str | None = None, chunk_strategy: ChunkStrategy | str | None = None, vector_backend: str | None = None) -> None:
        self.service_tier = RETRIEVAL_TIER
        self.mode: RetrievalMode = (mode or resolve_service_tier(self.service_tier, RETRIEVAL_MODE))  # type: ignore[assignment]
        self.chunk_strategy: ChunkStrategy = (chunk_strategy or CHUNK_STRATEGY)  # type: ignore[assignment]
        if self.mode not in {"bm25", "dense", "dense_page", "hybrid", "hybrid_rerank", "source_fusion"}: raise ValueError(f"Unknown retrieval mode: {self.mode}")
        self.documents = registry.documents(); self.chunks = build_chunks(self.documents, self.chunk_strategy)
        self._chunk_by_id = {chunk.chunk_id: index for index, chunk in enumerate(self.chunks)}
        self.bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks]) if self.chunks else None
        self.vectors: np.ndarray | None = None
        self.vector_backend_requested = (vector_backend or VECTOR_BACKEND).lower()
        self.vector_store: VectorStore | None = None
        self.vector_backend_status: dict[str, Any] = {"backend": "not_used", "status": "not_used", "count": 0}
        self._source_summary_vectors: np.ndarray | None = None
        self._source_summary_ids: list[str] = []
        self._source_summaries: dict[str, str] = {}
        self._source_summary_cache_status = "not_requested"
        if self.mode != "bm25":
            self.vectors = self._load_or_build_vectors()
            self._initialize_vector_store()
        if self.mode == "source_fusion":
            self._initialize_source_summary_index()

    def _cache_paths(self) -> tuple[Path, Path]:
        fingerprint = corpus_fingerprint(self.chunks)
        root = ROOT / "data" / "index"; root.mkdir(parents=True, exist_ok=True)
        return root / f"bge_m3_{self.chunk_strategy}_{fingerprint}.npy", root / f"bge_m3_{self.chunk_strategy}_{fingerprint}.json"

    def _load_or_build_vectors(self) -> np.ndarray:
        array_path, metadata_path = self._cache_paths()
        # Keep a per-content embedding cache so a document update encodes only
        # new/changed children; the corpus-level file remains a fast startup cache.
        root = ROOT / "data" / "index" / "chunk_embeddings"; root.mkdir(parents=True, exist_ok=True)
        paths = [root / f"{chunk_content_hash(chunk)}.npy" for chunk in self.chunks]
        if array_path.exists() and metadata_path.exists():
            vectors = np.load(array_path)
            # Migrate prior corpus caches without sending a second embedding
            # request. Subsequent document updates then reuse these rows.
            if len(vectors) == len(paths):
                for path, vector in zip(paths, vectors, strict=True):
                    if not path.exists(): np.save(path, vector)
            return vectors
        missing = [index for index, path in enumerate(paths) if not path.exists()]
        if missing:
            result = _embedder().encode([self.chunks[index].text for index in missing], return_dense=True, return_sparse=False, return_colbert_vecs=False)
            fresh = np.asarray(result["dense_vecs"], dtype=np.float32)
            for index, vector in zip(missing, fresh, strict=True): np.save(paths[index], vector)
        vectors = np.stack([np.load(path) for path in paths]).astype(np.float32) if paths else np.empty((0, 0), dtype=np.float32)
        np.save(array_path, vectors); metadata_path.write_text(json.dumps({"model": str(EMBEDDING_MODEL_PATH), "chunk_strategy": self.chunk_strategy, "count": len(self.chunks), "incremental_encoded": len(missing)}, ensure_ascii=False), encoding="utf-8")
        return vectors

    def _initialize_vector_store(self) -> None:
        if self.vectors is None:
            return
        local = LocalNpyVectorStore(self.chunk_strategy)
        local_status = local.sync(self.chunks, self.vectors)
        if self.vector_backend_requested != "milvus":
            self.vector_store, self.vector_backend_status = local, local_status
            return
        try:
            milvus = MilvusVectorStore()
            self.vector_backend_status = milvus.sync(self.chunks, self.vectors)
            self.vector_store = milvus
        except Exception as exc:
            self.vector_store = local
            self.vector_backend_status = {**local_status, "requested_backend": "milvus", "fallback_reason": f"{type(exc).__name__}: {exc}"}

    def _dense_ranking(self, query: str, source_types: set[str] | None = None) -> list[int]:
        if self.vectors is None or self.vector_store is None: return []
        with acquire_inference("embedding", uses_gpu=_DEVICE_STATUS.get("embedding") == "cuda") as waits:
            query_vector = np.asarray(_embedder().encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"][0], dtype=np.float32)
        self._embedding_wait_ms = waits["model_wait_ms"]
        self._embedding_gpu_wait_ms = waits["gpu_wait_ms"]
        self._last_query_vector = query_vector
        hits = self.vector_store.search(query_vector, limit=min(max(100, len(self.chunks)), len(self.chunks)), source_types=source_types)
        return [self._chunk_by_id[chunk_id] for chunk_id, _score in hits if chunk_id in self._chunk_by_id]

    def _source_summary_cache_paths(self) -> tuple[Path, Path]:
        fingerprint = corpus_fingerprint(self.chunks)
        root = ROOT / "data" / "index"; root.mkdir(parents=True, exist_ok=True)
        return root / f"source_summaries_{self.chunk_strategy}_{fingerprint}.npy", root / f"source_summaries_{self.chunk_strategy}_{fingerprint}.json"

    def _initialize_source_summary_index(self) -> None:
        """Build/load source summaries once per corpus, never per request."""
        vector_path, metadata_path = self._source_summary_cache_paths()
        grouped: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in self.chunks:
            grouped[self._source_identity(chunk)].append(chunk)
        self._source_summary_ids = sorted(grouped)
        self._source_summaries = {}
        for source_id in self._source_summary_ids:
            chunk = grouped[source_id][0]
            metadata = chunk.metadata
            self._source_summaries[source_id] = " ".join(str(part) for part in [
                metadata.get("title", ""), " ".join(map(str, metadata.get("topics", []))),
                " ".join(map(str, metadata.get("aliases", []))), metadata.get("document_kind", chunk.source_type), chunk.text[:1000],
            ])
        if vector_path.exists() and metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            vectors = np.load(vector_path)
            if payload.get("source_ids") == self._source_summary_ids and len(vectors) == len(self._source_summary_ids):
                self._source_summary_vectors = vectors.astype(np.float32)
                self._source_summary_cache_status = "hit"
                return
        with acquire_inference("embedding", uses_gpu=_DEVICE_STATUS.get("embedding") == "cuda"):
            result = _embedder().encode([self._source_summaries[source] for source in self._source_summary_ids], return_dense=True, return_sparse=False, return_colbert_vecs=False)
        self._source_summary_vectors = np.asarray(result["dense_vecs"], dtype=np.float32)
        np.save(vector_path, self._source_summary_vectors)
        metadata_path.write_text(json.dumps({"source_ids": self._source_summary_ids, "count": len(self._source_summary_ids), "chunk_strategy": self.chunk_strategy, "source_summary_schema": "title_topics_aliases_kind_first_fragment.v1"}, ensure_ascii=False), encoding="utf-8")
        self._source_summary_cache_status = "built"

    def _source_summary_ranking(self, query: str, candidate_sources: set[str]) -> dict[str, int]:
        """Return semantic source ranks for a non-destructive fusion experiment.

        The source summaries use manifest metadata plus the first page fragment.
        They can boost an already retrieved page but never remove a full-corpus
        page candidate, avoiding the recall loss of profile hard filtering.
        """
        if not self.chunks:
            return {}
        if self._source_summary_vectors is None:
            self._initialize_source_summary_index()
        # ``_dense_ranking`` has already embedded this exact query.  Reuse it
        # so source-summary fusion is a ranking experiment, not two model calls.
        query_vector = getattr(self, "_last_query_vector", None)
        if query_vector is None:
            query_vector = np.asarray(_embedder().encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"][0], dtype=np.float32)
        candidate_indexes = [index for index, source in enumerate(self._source_summary_ids) if source in candidate_sources]
        if not candidate_indexes:
            return {}
        scores = self._source_summary_vectors[candidate_indexes] @ query_vector
        order = np.argsort(-scores)
        return {self._source_summary_ids[candidate_indexes[index]]: rank for rank, index in enumerate(order, 1)}

    def _source_fusion(self, query: str, dense: list[int]) -> tuple[list[int], dict[int, dict[str, int]]]:
        ranks = self._source_summary_ranking(query, {self._source_identity(self.chunks[index]) for index in dense})
        # Rank fusion, rather than a source filter: the full corpus still
        # competes and an irrelevant source summary cannot hide a good page.
        details = {index: {"dense_rank": rank, "source_summary_rank": ranks.get(self._source_identity(self.chunks[index]), 10_000)} for rank, index in enumerate(dense, 1)}
        score = {index: 1 / (60 + detail["dense_rank"]) + 0.35 / (60 + detail["source_summary_rank"]) for index, detail in details.items()}
        return sorted(dense, key=lambda index: score[index], reverse=True), details

    def _bm25_ranking(self, query: str) -> list[int]:
        if self.bm25 is None: return []
        return np.argsort(-np.asarray(self.bm25.get_scores(tokenize(query)))).tolist()

    @staticmethod
    def _rrf(*rankings: list[int], k: int = 60) -> tuple[list[int], dict[int, dict[str, int]]]:
        scores: dict[int, float] = defaultdict(float); details: dict[int, dict[str, int]] = defaultdict(dict)
        for label, ranking in zip(("dense_rank", "bm25_rank"), rankings):
            for rank, index in enumerate(ranking, 1): scores[index] += 1 / (k + rank); details[index][label] = rank
        return sorted(scores, key=scores.get, reverse=True), details

    @staticmethod
    def infer_source_types(query: str) -> set[str]:
        lowered = query.lower()
        mapping = {
            "run_log": ("日志", "失败", "开始时间", "运行记录"),
            "run_status": ("状态", "耗时", "return_code"),
            "input_deck": ("输入", "deck", "边界", "语法", "设置"),
            "report": ("报告", "总结", "机理", "解释", "基线", "排名"),
            "script": ("脚本", "命名", "批量运行"),
            # These are source-profile concepts, not question IDs: they make
            # external thermal-protection literature discoverable when users
            # describe an experimental condition instead of naming a paper.
            "paper": ("论文", "文献", "研究", "机理", "tufroc", "热防护", "烧蚀", "热流", "剪切", "瓦片", "电弧喷流", "试件"),
            "scan_report": ("扫描", "ocr", "多层壁", "multiwall", "专利", "hypervelocity", "space shuttle", "再入", "飞行器"),
        }
        return {source_type for source_type, terms in mapping.items() if any(term in lowered for term in terms)}

    def _allowed_indices(self, source_types: Iterable[str] | None) -> list[int]:
        selected = set(source_types or [])
        return [index for index, chunk in enumerate(self.chunks) if not selected or chunk.source_type in selected]

    def _source_identity(self, chunk: Chunk) -> str:
        return str(chunk.metadata.get("source_id") or f"{chunk.source_type}:{Path(chunk.source_path).name}")

    @staticmethod
    def _profile_score(query: str, chunk: Chunk) -> int:
        """Match an optional source profile without polluting page text ranks."""
        lowered = query.lower()
        labels = [str(chunk.metadata.get("title", "")), *map(str, chunk.metadata.get("topics", [])), *map(str, chunk.metadata.get("aliases", []))]
        normalized = {label.strip().lower() for label in labels if len(label.strip()) >= 3}
        full_matches = {label for label in normalized if label in lowered}
        english_tokens = {token for label in normalized for token in re.findall(r"[a-z0-9]{4,}", label)}
        score = 4 * len(full_matches)
        for token in english_tokens:
            if token in lowered:
                score += 1
            # English titles are frequently queried by a distinctive token
            # rather than their complete formal title (e.g. "Advanced").
        return score


    def _profile_rerank(self, query: str, ranking: list[int]) -> list[int]:
        ordered = enumerate(ranking)
        return [index for _position, index in sorted(ordered, key=lambda item: (-self._profile_score(query, self.chunks[item[1]]), item[0]))]

    def _balanced_candidates(self, ranking: list[int], candidate_k: int, per_source: int = 3) -> list[int]:
        """Avoid one long PDF or repeated deck consuming the whole candidate pool."""
        selected: list[int] = []
        counts: dict[str, int] = defaultdict(int)
        for index in ranking:
            source = self._source_identity(self.chunks[index])
            if counts[source] >= per_source:
                continue
            selected.append(index); counts[source] += 1
            if len(selected) >= candidate_k:
                break
        return selected

    def _dense_sources_bm25_pages(self, dense: list[int], bm25_raw: list[int], candidate_k: int) -> tuple[list[int], dict[int, dict[str, int]]]:
        """Keep semantic source recall while using lexical evidence for page location.

        Dense vectors are comparatively robust to multilingual scientific
        paraphrases, while BM25 is often better at locating a named symbol,
        table heading or method term inside a long selected PDF.  This method
        applies the latter only after source selection; it is not a query-ID
        lookup or a source-specific rule.
        """
        bm25_by_source: dict[str, tuple[int, int]] = {}
        for rank, index in enumerate(bm25_raw, 1):
            bm25_by_source.setdefault(self._source_identity(self.chunks[index]), (index, rank))
        ranking: list[int] = []
        details: dict[int, dict[str, int]] = {}
        seen_sources: set[str] = set()
        for dense_rank, index in enumerate(dense, 1):
            source = self._source_identity(self.chunks[index])
            if source in seen_sources:
                continue
            seen_sources.add(source)
            page_index, bm25_rank = bm25_by_source.get(source, (index, 0))
            ranking.append(page_index)
            details[page_index] = {"dense_source_rank": dense_rank, "bm25_page_rank": bm25_rank}
            if len(ranking) >= candidate_k:
                break
        return ranking, details

    def retrieve(self, query: str, limit: int = 5, candidate_k: int = 20, source_types: Iterable[str] | None = None) -> list[EvidenceCard]:
        if not self.chunks: return []
        filters = set(source_types) if source_types is not None else self.infer_source_types(query)
        allowed = set(self._allowed_indices(filters))
        started = time.perf_counter()
        dense_raw = self._profile_rerank(query, [index for index in self._dense_ranking(query, filters) if index in allowed]) if self.mode != "bm25" else []
        bm25_raw = self._profile_rerank(query, [index for index in self._bm25_ranking(query) if index in allowed]) if self.mode not in {"dense", "source_fusion"} else []
        dense = self._balanced_candidates(dense_raw, candidate_k) if dense_raw else []
        bm25 = self._balanced_candidates(bm25_raw, candidate_k) if bm25_raw else []
        if self.mode == "dense": ranking, details = dense, {idx: {"dense_rank": rank} for rank, idx in enumerate(dense, 1)}
        elif self.mode == "source_fusion": ranking, details = self._source_fusion(query, dense)
        elif self.mode == "bm25": ranking, details = bm25, {idx: {"bm25_rank": rank} for rank, idx in enumerate(bm25, 1)}
        elif self.mode == "dense_page": ranking, details = self._dense_sources_bm25_pages(dense, bm25_raw, candidate_k)
        else: ranking, details = self._rrf(dense, bm25)
        ranking = ranking[:candidate_k]
        rerank_scores: dict[int, float] = {}
        if self.mode == "hybrid_rerank" and ranking:
            rerank_candidates: list[int] = []
            candidate_sources: set[str] = set()
            for index in ranking:
                source = self._source_identity(self.chunks[index])
                if source not in candidate_sources:
                    rerank_candidates.append(index)
                    candidate_sources.add(source)
                if len(rerank_candidates) == 8:
                    break
            with acquire_inference("reranker", uses_gpu=_DEVICE_STATUS.get("reranker") == "cuda") as waits:
                scores = _rerank_scores(query, [self.chunks[index].text for index in rerank_candidates])
            self._reranker_wait_ms = waits["model_wait_ms"]
            self._reranker_gpu_wait_ms = waits["gpu_wait_ms"]
            rerank_scores = dict(zip(rerank_candidates, map(float, scores), strict=True))
            ranking = sorted(rerank_candidates, key=lambda index: rerank_scores[index], reverse=True)
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        cards = []
        seen_sources: set[str] = set()
        # A citation list should cover independent sources. Returning five
        # fragments from one deck looks confident but gives the agent little
        # additional evidence and hides logs or reports lower in the ranking.
        for index in ranking:
            chunk = self.chunks[index]
            identity = self._citation_identity(chunk)
            if identity in seen_sources:
                continue
            seen_sources.add(identity)
            rank = len(cards) + 1
            score = rerank_scores.get(index, 1 / (60 + rank))
            cards.append(EvidenceCard(claim=f"检索到与问题相关的{chunk.source_type}证据。", source_type=chunk.source_type, source_path=chunk.source_path, excerpt=chunk.text.replace("\n", " ")[:850], score=round(float(score), 6), retrieval={"mode": self.mode, "service_tier": self.service_tier, "chunk_strategy": self.chunk_strategy, "vector_backend": self.vector_backend_status.get("backend", "not_used"), "vector_backend_fallback": self.vector_backend_status.get("fallback_reason", ""), "source_type_filter": sorted(filters), "chunk_id": chunk.chunk_id, "rank": rank, "start_line": chunk.start_line, "end_line": chunk.end_line, "citation_identity": identity, "authority": self._authority(chunk), **chunk.metadata, **details.get(index, {}), "rerank_score": rerank_scores.get(index), "embedding_queue_wait_ms": getattr(self, "_embedding_wait_ms", 0.0), "embedding_gpu_wait_ms": getattr(self, "_embedding_gpu_wait_ms", 0.0), "reranker_queue_wait_ms": getattr(self, "_reranker_wait_ms", 0.0), "reranker_gpu_wait_ms": getattr(self, "_reranker_gpu_wait_ms", 0.0), "latency_ms": elapsed}))
            if len(cards) >= limit:
                break
        return cards

    @staticmethod
    def _citation_identity(chunk: Chunk) -> str:
        # Native project files cite at file granularity; external PDFs cite page.
        if chunk.metadata.get("page") is not None:
            return f"{chunk.source_path}#page={chunk.metadata['page']}"
        # Repeated case folders often contain byte-identical template decks.
        # A basename identity prevents them crowding out reports in a citation set.
        return f"{chunk.source_type}:{Path(chunk.source_path).name}"

    @staticmethod
    def _authority(chunk: Chunk) -> str:
        """External metadata wins; project operational evidence is classified by type."""
        if chunk.metadata.get("authority"):
            return str(chunk.metadata["authority"])
        return "B" if chunk.source_type == "report" else "A"

    def search(self, query: str, limit: int = 5, source_types: Iterable[str] | None = None) -> list[EvidenceCard]: return self.retrieve(query, limit, source_types=source_types)

    def vector_status(self) -> dict[str, Any]:
        return {**self.vector_backend_status, "model_devices": dict(_DEVICE_STATUS), "source_summary_index": {"count": len(self._source_summary_ids), "cache": self._source_summary_cache_status}}
