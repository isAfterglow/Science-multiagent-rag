"""Fast, traceable PDF document IR built from PyMuPDF blocks and tables."""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz


@dataclass
class DocumentBlock:
    block_id: str
    page: int
    block_type: str
    text: str
    bbox: list[float]
    reading_order: int
    section: str
    parser: str
    confidence: float | None = None
    table_csv: str = ""
    table_confidence: float | None = None
    column: int = 0


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\u00a0", " ")).strip()


def _table_payload(rows: list[list[Any]]) -> tuple[str, str, float]:
    cleaned = [[_clean(str(cell or "")) for cell in row] for row in rows if any(_clean(str(cell or "")) for cell in row)]
    if not cleaned:
        return "", "", 0.0
    width = max(len(row) for row in cleaned)
    padded = [row + [""] * (width - len(row)) for row in cleaned]
    header = padded[0]
    markdown = "| " + " | ".join(header) + " |\n| " + " | ".join(["---"] * width) + " |"
    markdown += "".join("\n| " + " | ".join(row) + " |" for row in padded[1:])
    buffer = io.StringIO(); writer = csv.writer(buffer); writer.writerows(padded)
    nonempty = sum(bool(cell) for row in padded for cell in row)
    confidence = 0.9 if len(padded) >= 3 and width >= 2 and nonempty >= 8 else 0.65 if len(padded) >= 2 and width >= 2 and nonempty >= 5 else 0.0
    return markdown, buffer.getvalue(), confidence


def _table_blocks(page: fitz.Page, page_number: int, section: str) -> list[DocumentBlock]:
    try:
        tables = page.find_tables().tables
    except Exception:
        return []
    output: list[DocumentBlock] = []
    for index, table in enumerate(tables):
        markdown, csv_text, confidence = _table_payload(table.extract())
        if not markdown:
            continue
        output.append(DocumentBlock(
            block_id=f"p{page_number}:table:{index}", page=page_number, block_type="table", text=markdown,
            bbox=[round(float(value), 2) for value in table.bbox], reading_order=0, section=section,
            parser="pymupdf_find_tables", table_csv=csv_text, table_confidence=confidence,
        ))
    return output


def _intersects(left: list[float], right: list[float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def _is_heading(text: str, bbox: list[float], page_height: float) -> bool:
    compact = _clean(text)
    if len(compact) < 3 or len(compact) > 150:
        return False
    if re.match(r"^(?:[IVXLC]+|\d+(?:\.\d+)*)[.\s]+[A-Z]", compact):
        return True
    return bbox[1] < page_height * 0.18 and compact.upper() == compact and any(char.isalpha() for char in compact)


def _ordered_text_blocks(page: fitz.Page, page_number: int, ignored: set[str], section: str) -> tuple[list[DocumentBlock], int]:
    raw: list[DocumentBlock] = []
    height, width = page.rect.height, page.rect.width
    for index, item in enumerate(page.get_text("blocks")):
        x0, y0, x1, y1, text, _block_no, block_kind = item[:7]
        text = _clean(text)
        normalized = re.sub(r"\s+", " ", text).lower()
        if block_kind != 0 or len(text) < 2 or normalized in ignored:
            continue
        # Remaining marginal page numbers are never useful retrieval evidence.
        if (y0 < height * 0.06 or y1 > height * 0.94) and len(text) < 80:
            continue
        bbox = [round(float(value), 2) for value in (x0, y0, x1, y1)]
        kind = "heading" if _is_heading(text, bbox, height) else "paragraph"
        raw.append(DocumentBlock(f"p{page_number}:text:{index}", page_number, kind, text, bbox, 0, section, "pymupdf_blocks"))
    left = [block for block in raw if (block.bbox[0] + block.bbox[2]) / 2 < width * 0.48 and (block.bbox[2] - block.bbox[0]) < width * 0.62]
    right = [block for block in raw if (block.bbox[0] + block.bbox[2]) / 2 >= width * 0.48 and (block.bbox[2] - block.bbox[0]) < width * 0.62]
    columns = 2 if len(left) >= 2 and len(right) >= 2 else 1
    if columns == 2:
        for block in left: block.column = 1
        for block in right: block.column = 2
        wide = [block for block in raw if block not in left and block not in right]
        # Full-width headings precede the two-column body; within a column,
        # vertical order is stable and avoids alternating left/right lines.
        ordered = sorted(wide, key=lambda block: block.bbox[1]) + sorted(left, key=lambda block: block.bbox[1]) + sorted(right, key=lambda block: block.bbox[1])
    else:
        ordered = sorted(raw, key=lambda block: (block.bbox[1], block.bbox[0]))
    for order, block in enumerate(ordered, 1):
        block.reading_order = order
    return ordered, columns


def parse_pdf(path: Path, source_id: str, *, ocr_pages: dict[int, tuple[str, float | None, str | None]] | None = None) -> dict[str, Any]:
    """Produce a serializable page/block IR without invoking heavyweight layout models."""
    pdf = fitz.open(path)
    marginal = Counter()
    for page in pdf:
        for item in page.get_text("blocks"):
            text = _clean(str(item[4]))
            if len(text) < 80:
                marginal[re.sub(r"\s+", " ", text).lower()] += 1
    ignored = {text for text, count in marginal.items() if count >= 3 and text}
    pages: list[dict[str, Any]] = []
    current_section = ""
    for number, page in enumerate(pdf, 1):
        if ocr_pages and number in ocr_pages:
            text, confidence, _image = ocr_pages[number]
            if text.strip():
                blocks = [DocumentBlock(f"p{number}:ocr:0", number, "ocr_paragraph", text, [0, 0, float(page.rect.width), float(page.rect.height)], 1, current_section, "rapidocr", confidence)]
                pages.append({"page": number, "width": round(page.rect.width, 2), "height": round(page.rect.height, 2), "column_count": 1,
                              "blocks": [asdict(block) for block in blocks]})
                continue
        text_blocks, columns = _ordered_text_blocks(page, number, ignored, current_section)
        for block in text_blocks:
            if block.block_type == "heading":
                current_section = block.text
            block.section = current_section
        tables = _table_blocks(page, number, current_section)
        retained_text = [block for block in text_blocks if not any(_intersects(block.bbox, table.bbox) for table in tables)]
        blocks = retained_text + tables
        blocks.sort(key=lambda block: (block.reading_order or 10_000, block.bbox[1], block.bbox[0]))
        for order, block in enumerate(blocks, 1):
            block.reading_order = order
        pages.append({"page": number, "width": round(page.rect.width, 2), "height": round(page.rect.height, 2), "column_count": columns,
                      "blocks": [asdict(block) for block in blocks]})
    return {"source_id": source_id, "parser": "pymupdf_blocks", "page_count": len(pdf), "pages": pages}
