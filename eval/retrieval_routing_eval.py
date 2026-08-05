"""Evaluate a conservative lexical-confidence router against full-corpus Dense.

The router is deliberately an experiment: it never changes production default
until both overall and tagged-slice quality gates pass.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.config import ROOT
from eval.scientific_multimodal_retrieval_eval import _summary, evaluate


def lexical_confidence(question: str) -> tuple[bool, dict]:
    lowered = question.lower()
    technical = re.findall(r"\b[a-z][a-z0-9_/-]{3,}\b", lowered)
    explicit_shape = bool(re.search(r"\b(?:table|variables?|equation|figure|mole fractions|w/cm2|kw)\b|\d+\s*(?:w/cm2|kw|kpa|mm)\b", lowered))
    # Require both an exact technical form and at least two stable Latin tokens;
    # generic Chinese words such as '表格' must not replace semantic retrieval.
    return explicit_shape and len(set(technical)) >= 2, {"technical_tokens": sorted(set(technical)), "explicit_shape": explicit_shape}


def run() -> dict:
    dense = evaluate("dense", "parent_child")
    bm25 = evaluate("bm25", "parent_child")
    dense_rows = {row["id"]: row for row in dense["results"]}
    bm25_rows = {row["id"]: row for row in bm25["results"]}
    selected, decisions = [], []
    for row in dense["results"]:
        use_bm25, features = lexical_confidence(row["question"])
        chosen = dict(bm25_rows[row["id"]] if use_bm25 else row)
        chosen["route"] = "bm25" if use_bm25 else "dense"
        selected.append(chosen)
        decisions.append({"id": row["id"], "route": chosen["route"], "features": features})
    return {"strategy": "lexical_confidence_v1", "route_counts": {name: sum(item["route"] == name for item in decisions) for name in ("dense", "bm25")}, "metrics": _summary(selected), "dense_baseline": dense["metrics"], "bm25_baseline": bm25["metrics"], "decisions": decisions, "results": selected}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=ROOT / "reports" / "retrieval_routing_eval.json")
    args = parser.parse_args(); result = run(); args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"route_counts": result["route_counts"], "metrics": result["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
