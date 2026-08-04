"""Structural quality checks for persisted PDF DocumentIR artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from app.config import ROOT

PARSED = ROOT / "data" / "knowledge_sources" / "parsed"


def run() -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(PARSED.glob("*.document_ir.json"))]
    pages = [page for report in reports for page in report["pages"]]
    blocks = [block for page in pages for block in page["blocks"]]
    ordered = []
    for page in pages:
        by_column: dict[int, list[dict]] = {}
        for block in page["blocks"]:
            by_column.setdefault(block.get("column", 0), []).append(block)
        ordered.append(all(items == sorted(items, key=lambda item: item["reading_order"]) for items in by_column.values()))
    tables = [block for block in blocks if block["block_type"] == "table"]
    result = {
        "sources": len(reports), "pages": len(pages), "blocks": len(blocks),
        "parse_coverage": round(sum(bool(page["blocks"]) for page in pages) / len(pages), 4) if pages else 0,
        "two_column_pages": sum(page["column_count"] == 2 for page in pages),
        "reading_order_valid_rate": round(sum(ordered) / len(ordered), 4) if ordered else 0,
        "tables": len(tables), "high_confidence_tables": sum((block.get("table_confidence") or 0) >= 0.85 for block in tables),
        "table_candidates": sum((block.get("table_confidence") or 0) < 0.85 for block in tables),
        "ocr_blocks": sum(block["block_type"] == "ocr_paragraph" for block in blocks),
    }
    output = ROOT / "reports" / "document_ir_eval.json"; output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
