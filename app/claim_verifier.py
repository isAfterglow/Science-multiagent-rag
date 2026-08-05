"""Deterministic Claim-Evidence verification for final research statements.

This is intentionally a provenance/authority verifier, not an unvalidated NLI
model. A statement is only ``supported`` when its declared source can be
resolved to the exact Registry result or raw EvidenceCard used to create it.
"""
from __future__ import annotations

import re
from typing import Any

from app.models import ClaimVerification, GroundedStatement


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _card_ref(card: dict[str, Any]) -> dict[str, str]:
    retrieval = card.get("retrieval", {})
    return {
        "source_path": str(card.get("source_path", "")),
        "chunk_id": str(retrieval.get("chunk_id", "")),
        "authority": str(retrieval.get("authority", "")),
    }


def _numeric_table_is_locatable(statement: GroundedStatement, cards: list[dict[str, Any]]) -> bool:
    """Numeric table claims require a table block and original PDF page."""
    if not re.search(r"\d", statement.support) or "table" not in statement.source_path.lower():
        return True
    return any(card.get("retrieval", {}).get("block_type") == "table" and card.get("retrieval", {}).get("page") is not None for card in cards)


def verify_grounded_statements(
    statements: list[dict[str, Any]],
    cards: list[dict[str, Any]],
    analysis_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve every displayed statement to exact evidence or a safe boundary."""
    verdicts: list[dict[str, Any]] = []
    for index, payload in enumerate(statements):
        statement = GroundedStatement.model_validate(payload)
        if statement.evidence_kind == "limitation" or statement.source_path in {"critic", "planner_agent"}:
            verdict = ClaimVerification(statement_index=index, status="context_only", reasons=["限制、规划建议或审批提示不作为外部事实结论。"])
        elif statement.source_path == "simulation_plan_draft":
            verdict = ClaimVerification(statement_index=index, status="context_only", reasons=["SimulationPlan 是待审批草案，不是新产生的仿真事实。"])
        elif statement.evidence_kind == "analysis":
            matching = [item for item in analysis_evidence if str(item.get("source", "")) == statement.source_path]
            if matching and any(_compact(statement.support) == _compact(str(item.get("claim", ""))) for item in matching):
                verdict = ClaimVerification(statement_index=index, status="supported", evidence_refs=[{"source_path": statement.source_path, "kind": "registry_analysis"}], reasons=["定量结论绑定到同一 Registry 工具输出。"])
            else:
                verdict = ClaimVerification(statement_index=index, status="insufficient", reasons=["未找到与该定量结论匹配的 Registry 分析证据。"])
        else:
            matching = [card for card in cards if str(card.get("source_path", "")) == statement.source_path and (not statement.chunk_id or str(card.get("retrieval", {}).get("chunk_id", "")) == statement.chunk_id)]
            polarities = {str(card.get("retrieval", {}).get("claim_polarity", "supports")) for card in matching}
            if matching and "refutes" in polarities:
                verdict = ClaimVerification(statement_index=index, status="conflicted", evidence_refs=[_card_ref(card) for card in matching], reasons=["引用证据被标记为与该 Claim 冲突。"])
            elif matching and not _numeric_table_is_locatable(statement, matching):
                verdict = ClaimVerification(statement_index=index, status="insufficient", evidence_refs=[_card_ref(card) for card in matching], reasons=["表格数值 Claim 缺少可回到原 PDF 页的 table block 元数据。"])
            elif matching and any(_compact(statement.support) in _compact(str(card.get("excerpt", ""))) for card in matching):
                verdict = ClaimVerification(statement_index=index, status="supported", evidence_refs=[_card_ref(card) for card in matching], reasons=["文档 Claim 可回溯到同一原始摘录和 chunk。"])
            else:
                verdict = ClaimVerification(statement_index=index, status="insufficient", evidence_refs=[_card_ref(card) for card in matching], reasons=["文档 Claim 缺少可定位的原始摘录支持。"])
        verdicts.append(verdict.model_dump())
    return verdicts
