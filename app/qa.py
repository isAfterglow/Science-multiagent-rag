"""Stage-one deterministic router for cited baseline answers, not an LLM agent."""
from __future__ import annotations

import re

from app.analysis import case_detail, parameter_correlation, top_cases
from app.registry import SimulationRegistry
from app.retrieval import HybridRetriever

METRICS = {"early_1_2_rmse", "early_1mm_rmse", "early_2mm_rmse", "temp_mean_rmse", "rmse_Tsh", "rmse_T1mmh", "rmse_T2mmh", "rmse_T4mmh", "rmse_T8mmh", "rmse_T16mmh", "rmse_T24mmh"}

def _explicit_metric(question: str) -> str | None:
    for metric in METRICS:
        if metric.lower() in question.lower(): return metric
    return None

def answer(question: str, registry: SimulationRegistry, retrieval_mode: str | None = None) -> dict:
    explicit_metric = _explicit_metric(question)
    metric = explicit_metric or "early_1_2_rmse"
    case_match = re.search(r"case[_ -]?(\d+)", question, re.I)
    if case_match:
        case_id = f"case_{int(case_match.group(1)):03d}"
        return {"route": "case_detail", "answer": case_detail(registry, case_id), "citations": ["simulation_registry"]}
    if explicit_metric and any(word in question.lower() for word in ("相关", "correlation", "影响", "敏感", "关系")):
        return {"route": "parameter_correlation", "answer": parameter_correlation(registry, metric), "citations": ["simulation_registry"]}
    if any(word in question.lower() for word in ("最好", "最优", "top", "最低")):
        return {"route": "top_cases", "answer": top_cases(registry, metric), "citations": ["simulation_registry"]}
    cards = HybridRetriever(registry, mode=retrieval_mode).search(question)
    return {"route": "hybrid_retrieval", "answer": [card.model_dump() for card in cards], "citations": [card.source_path for card in cards]}
