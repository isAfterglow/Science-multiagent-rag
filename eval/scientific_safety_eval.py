"""Regression harness for scientific evidence authority and OCR safeguards."""
from __future__ import annotations

import json
from pathlib import Path

from app.config import ROOT
from app.evidence_policy import review_evidence_policy


def _card(authority: str, *, ocr: bool = False, confidence: float | None = None) -> dict:
    return {"retrieval": {"authority": authority, "ocr_used": ocr, "ocr_confidence": confidence}}


CASES = [
    {"id": "safe-01", "question": "项目历史仿真的 early_1_2_rmse 是多少？", "cards": [_card("C")], "analysis": [], "blocked": True},
    {"id": "safe-02", "question": "项目运行耗时是多少？", "cards": [_card("A")], "analysis": [], "blocked": False},
    {"id": "safe-03", "question": "项目参数影响是什么？", "cards": [_card("C")], "analysis": [{"source": "simulation_registry"}], "blocked": False},
    {"id": "safe-04", "question": "论文对烧蚀机理的解释是什么？", "cards": [_card("C")], "analysis": [], "blocked": False},
    {"id": "safe-05", "question": "扫描图中的精确厚度是多少？", "cards": [_card("D", ocr=True, confidence=0.62)], "analysis": [], "blocked": True},
    {"id": "safe-06", "question": "扫描资料中的专利号是什么？", "cards": [_card("D", ocr=True, confidence=0.91)], "analysis": [], "blocked": False},
    {"id": "safe-07", "question": "项目真实 case 的指标是多少？", "cards": [_card("D", ocr=True, confidence=0.55)], "analysis": [], "blocked": True},
    {"id": "safe-08", "question": "外部热防护资料中的结构是什么？", "cards": [_card("D", ocr=True, confidence=0.55)], "analysis": [], "blocked": False},
    {"id": "safe-09", "question": "表格中 200-4000 Pa 的精确数值是多少？", "cards": [{"retrieval": {"authority": "C", "block_type": "table", "table_confidence": 0.65}}], "analysis": [], "blocked": True},
]


def run() -> dict:
    rows = []
    for case in CASES:
        reasons = review_evidence_policy(case["question"], case["cards"], case["analysis"])
        actual = bool(reasons)
        rows.append({"id": case["id"], "expected_blocked": case["blocked"], "actual_blocked": actual, "passed": actual == case["blocked"], "reasons": reasons})
    return {"questions": len(rows), "passed": sum(row["passed"] for row in rows), "pass_rate": sum(row["passed"] for row in rows) / len(rows), "results": rows}


if __name__ == "__main__":
    result = run(); path = ROOT / "reports" / "scientific_safety_eval.json"; path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(result, ensure_ascii=False, indent=2))
