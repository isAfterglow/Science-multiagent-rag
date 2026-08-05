"""Reproducible CPU/GPU retrieval benchmark on the registered scientific corpus.

Run one process per device so CUDA allocator state and model singletons do not
leak across measurements.  It deliberately reuses the production vector cache
for query latency, while the separate passage batch measures encode throughput.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import median, quantiles

from app.config import DB_PATH, ROOT
from app.registry import SimulationRegistry


def _p95(values: list[float]) -> float:
    return quantiles(values, n=20)[18] if len(values) >= 20 else max(values)


def run(device: str, queries: int, passages: int, include_reranker: bool) -> dict:
    # Config values are imported once, so set both module constants before the
    # first model construction.  This keeps benchmark invocation ergonomic.
    import app.retrieval as retrieval
    retrieval.EMBEDDING_DEVICE = device
    retrieval.RERANKER_DEVICE = device
    retriever = retrieval.HybridRetriever(SimulationRegistry(DB_PATH), mode="dense", chunk_strategy="parent_child")
    sample = [chunk.text for chunk in retriever.chunks[: min(passages, len(retriever.chunks))]]
    torch_info: dict[str, object] = {"cuda_available": False, "max_memory_allocated_bytes": 0}
    try:
        import torch
        torch_info["cuda_available"] = torch.cuda.is_available()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None  # type: ignore[assignment]
    started = time.perf_counter()
    encoded = retrieval._embedder().encode(sample, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    embedding_ms = (time.perf_counter() - started) * 1000
    prompts = [
        "MOOSE framework parallel execution thermal hydraulics",
        "arc jet heat flux measurement uncertainty",
        "扫描报告中的多层壁热防护结构",
        "PICA carbon phenolic pyrolysis material response",
        "table thermal conductivity units and test conditions",
    ]
    latencies: list[float] = []
    for index in range(queries):
        started = time.perf_counter()
        retriever.search(prompts[index % len(prompts)], limit=5)
        latencies.append((time.perf_counter() - started) * 1000)
    rerank_ms: float | None = None
    if include_reranker:
        started = time.perf_counter()
        retrieval._rerank_scores(prompts[0], sample[:8])
        rerank_ms = (time.perf_counter() - started) * 1000
    try:
        if device == "cuda" and torch is not None and torch.cuda.is_available():
            torch_info["max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            torch_info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        torch_info["memory_error"] = f"{type(exc).__name__}: {exc}"
    return {
        "device_requested": device,
        "device_actual": retriever.vector_status().get("model_devices", {}),
        "chunk_count": len(retriever.chunks),
        "passage_embedding": {"texts": len(sample), "elapsed_ms": round(embedding_ms, 3), "texts_per_second": round(len(sample) / (embedding_ms / 1000), 3), "dimensions": len(encoded["dense_vecs"][0]) if sample else 0},
        "dense_query": {"queries": queries, "p50_latency_ms": round(median(latencies), 3), "p95_latency_ms": round(_p95(latencies), 3), "mean_latency_ms": round(sum(latencies) / len(latencies), 3)},
        "reranker_8_candidates_ms": round(rerank_ms, 3) if rerank_ms is not None else None,
        "cuda": torch_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default=os.getenv("EMBEDDING_DEVICE", "cpu"))
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--passages", type=int, default=24)
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "retrieval_device_benchmark.json")
    args = parser.parse_args()
    result = run(args.device, args.queries, args.passages, args.reranker)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
