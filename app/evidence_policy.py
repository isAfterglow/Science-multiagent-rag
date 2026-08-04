"""Deterministic evidence policy used by the final scientific reviewer."""
from __future__ import annotations

import re
from typing import Any

PROJECT_FACT_TERMS = ("项目", "历史 case", "仿真", "指标", "rmse", "参数", "运行", "耗时", "return_code", "状态", "真实")
EXACT_VALUE_TERMS = ("精确", "数值", "尺寸", "厚度", "温度", "专利号", "多少", "几")
MIN_OCR_CONFIDENCE = 0.85


def review_evidence_policy(question: str, cards: list[dict[str, Any]], analysis_evidence: list[dict[str, Any]]) -> list[str]:
    """Return blocking reasons without allowing an LLM to weaken provenance rules."""
    lowered = question.lower()
    reasons: list[str] = []
    authorities = {str(card.get("retrieval", {}).get("authority", "")) for card in cards}
    asks_project_fact = any(term in lowered for term in PROJECT_FACT_TERMS)
    if asks_project_fact and not analysis_evidence and not (authorities & {"A", "B"}):
        reasons.append("项目实际事实缺少 A/B 级项目证据；公开论文或扫描件不能替代 Registry、日志、输入 deck 或项目报告。")
    asks_exact_value = bool(re.search(r"\d", question)) or any(term in lowered for term in EXACT_VALUE_TERMS)
    low_confidence_ocr = [card for card in cards if card.get("retrieval", {}).get("ocr_used") and (card.get("retrieval", {}).get("ocr_confidence") or 0) < MIN_OCR_CONFIDENCE]
    if asks_exact_value and low_confidence_ocr:
        reasons.append(f"精确数值请求命中 OCR 置信度低于 {MIN_OCR_CONFIDENCE:.2f} 的扫描证据；需人工核对原始页图后才能下结论。")
    low_confidence_tables = [card for card in cards if card.get("retrieval", {}).get("block_type") == "table" and (card.get("retrieval", {}).get("table_confidence") or 0) < MIN_OCR_CONFIDENCE]
    if asks_exact_value and low_confidence_tables:
        reasons.append(f"精确数值请求命中表格结构置信度低于 {MIN_OCR_CONFIDENCE:.2f} 的候选表格；需核对原始 PDF 页面。")
    return reasons
