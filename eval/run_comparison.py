"""Compare retrieval-only, single-tool-agent and evidence-driven multi-agent baselines."""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import quantiles

from app.multi_agent import run_multi_agent
from app.qa import answer
from app.registry import SimulationRegistry
from app.retrieval import HybridRetriever

ROOT = Path(__file__).parent

def _load() -> list[dict]:
    rows: list[dict] = []
    for path in (ROOT / "questions.jsonl", ROOT / "stage2_questions.jsonl"):
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return rows

def _score(kind: str, item: dict, result: dict) -> dict:
    if kind == "rag_only":
        citations = result.get("citations", [])
        has_analysis = False
        reviewed = False
    elif kind == "tool_agent":
        citations = result.get("citations", [])
        has_analysis = result.get("route") in {"top_cases", "parameter_correlation", "case_detail"}
        reviewed = False
    else:
        citations = [card["source_path"] for card in result.get("evidence_cards", [])] + [item["source"] for item in result.get("analysis_evidence", [])]
        has_analysis = bool(result.get("analysis_evidence"))
        reviewed = bool(result.get("review", {}).get("approved"))
    if item["type"] == "mixed": passed = bool(citations) and has_analysis and reviewed
    elif item["type"] == "retrieval": passed = bool(citations)
    else: passed = has_analysis and bool(citations)
    return {"passed": passed, "has_citation": bool(citations), "has_analysis": has_analysis, "reviewed": reviewed}

def _run(kind: str, question: str, registry: SimulationRegistry, retrieval_mode: str, rag_retriever: HybridRetriever) -> dict:
    if kind == "rag_only":
        cards = rag_retriever.search(question)
        return {"citations": [card.source_path for card in cards]}
    if kind in {"tool_agent", "single_agent_same_tools"}: return answer(question, registry, retrieval_mode)
    return run_multi_agent(question, registry, retrieval_mode=retrieval_mode)

def run(registry: SimulationRegistry, retrieval_mode: str = "bm25") -> dict:
    questions = _load(); variants = {}
    for kind in ("rag_only", "tool_agent", "single_agent_same_tools", "multi_agent", "multi_agent_no_critic", "multi_agent_no_reviewer"):
        rows = []; latencies = []
        rag_retriever = HybridRetriever(registry, mode=retrieval_mode)
        for item in questions:
            start = time.perf_counter(); result = _run(kind, item["question"], registry, retrieval_mode, rag_retriever); latency = (time.perf_counter() - start) * 1000
            if kind == "single_agent_same_tools":
                # Same retrieval/Registry capabilities as tool_agent, but no
                # specialist split, critic, synthesis or reviewer gate.
                score = _score("tool_agent", item, result)
            elif kind.startswith("multi_agent_no_"):
                full = result
                score = _score("multi_agent", item, full)
                if kind == "multi_agent_no_critic":
                    score["passed"] = score["has_citation"] and (item["type"] == "retrieval" or score["has_analysis"])
                    score["reviewed"] = False
                else:
                    score["passed"] = score["has_citation"] and (item["type"] == "retrieval" or score["has_analysis"])
                    score["reviewed"] = False
            else:
                score = _score(kind, item, result)
            rows.append({"id": item["id"], "type": item["type"], "latency_ms": round(latency, 3), **score}); latencies.append(latency)
        total = len(rows)
        variants[kind] = {"retrieval_mode": retrieval_mode, "total": total, "passed": sum(row["passed"] for row in rows), "pass_rate": round(sum(row["passed"] for row in rows) / total, 4), "citation_rate": round(sum(row["has_citation"] for row in rows) / total, 4), "analysis_coverage": round(sum(row["has_analysis"] for row in rows) / total, 4), "reviewed_rate": round(sum(row["reviewed"] for row in rows) / total, 4), "p95_latency_ms": round(quantiles(latencies, n=20)[18], 3), "results": rows}
    return variants
