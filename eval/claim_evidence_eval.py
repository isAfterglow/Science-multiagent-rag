"""Offline regression for Claim-Evidence provenance verdicts."""
from __future__ import annotations

import json
from pathlib import Path

from app.claim_verifier import verify_grounded_statements
from app.config import ROOT

CASES = Path(__file__).with_name("claim_evidence_cases.json")


def _payload(case: dict) -> tuple[dict, list[dict], list[dict]]:
    kind, support = case["kind"], case["support"]
    if kind in {"document", "mismatch", "conflict"}:
        excerpt = support if kind == "document" else "原始摘录支持。"
        statement = {"text": support, "evidence_kind": "document", "source_path": "report.md", "chunk_id": "chunk-1", "support": support}
        cards = [{"source_path": "report.md", "excerpt": excerpt, "retrieval": {"chunk_id": "chunk-1", "authority": "B", "claim_polarity": "refutes" if kind == "conflict" else "supports"}}]
        return statement, cards, []
    if kind in {"numeric_table", "numeric_table_missing_page"}:
        metadata = {"chunk_id": "table-1", "authority": "C", "block_type": "table"}
        if kind == "numeric_table": metadata["page"] = 4
        return {"text": support, "evidence_kind": "document", "source_path": "thermal_table.pdf", "chunk_id": "table-1", "support": support}, [{"source_path": "thermal_table.pdf", "excerpt": support, "retrieval": metadata}], []
    if kind == "cross_source_conflict":
        cards = [{"source_path": "source-a.pdf", "excerpt": support, "retrieval": {"chunk_id": "a", "authority": "C", "claim_polarity": "supports"}}, {"source_path": "source-b.pdf", "excerpt": support, "retrieval": {"chunk_id": "b", "authority": "C", "claim_polarity": "refutes"}}]
        return {"text": support, "evidence_kind": "document", "source_path": "source-b.pdf", "chunk_id": "b", "support": support}, cards, []
    if kind == "missing_chunk":
        return {"text": support, "evidence_kind": "document", "source_path": "missing.md", "chunk_id": "missing", "support": support}, [], []
    if kind in {"analysis", "bad_analysis"}:
        source = "simulation_registry" if kind == "analysis" else "unknown_tool"
        evidence = [{"source": "simulation_registry", "claim": support}] if kind == "analysis" else []
        return {"text": support, "evidence_kind": "analysis", "source_path": source, "support": support}, [], evidence
    if kind == "plan": return {"text": support, "evidence_kind": "analysis", "source_path": "simulation_plan_draft", "support": support}, [], []
    return {"text": support, "evidence_kind": "limitation", "source_path": "critic", "support": support}, [], []


def run() -> dict:
    rows = []
    for case in json.loads(CASES.read_text(encoding="utf-8")):
        statement, cards, analysis = _payload(case)
        verdict = verify_grounded_statements([statement], cards, analysis)[0]
        rows.append({"id": case["id"], "expected": case["expected"], "actual": verdict["status"], "passed": verdict["status"] == case["expected"], "reasons": verdict["reasons"]})
    by_status = {status: sum(row["actual"] == status for row in rows) for status in ("supported", "insufficient", "conflicted", "context_only")}
    numeric = [row for row in rows if row["id"] in {"ce-29", "ce-30"}]
    return {"questions": len(rows), "passed": sum(row["passed"] for row in rows), "pass_rate": sum(row["passed"] for row in rows) / len(rows), "status_counts": by_status, "numeric_table_page_verification_rate": sum(row["passed"] for row in numeric) / len(numeric), "results": rows}


if __name__ == "__main__":
    result = run(); output = ROOT / "reports" / "claim_evidence_eval.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, ensure_ascii=False, indent=2))
