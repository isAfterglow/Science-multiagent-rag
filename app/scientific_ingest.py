"""Traceable ingest for public scientific PDFs and scanned technical reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from urllib.request import Request, urlopen

import fitz

from app.config import DB_PATH, ROOT
from app.document_ir import parse_pdf
from app.registry import SimulationRegistry

SOURCES_ROOT = ROOT / "data" / "knowledge_sources"
VERSIONED_MANIFEST_PATH = ROOT / "knowledge_sources" / "manifest.json"
LEGACY_MANIFEST_PATH = SOURCES_ROOT / "manifest.json"
MANIFEST_PATH = VERSIONED_MANIFEST_PATH if VERSIONED_MANIFEST_PATH.exists() else LEGACY_MANIFEST_PATH
RAW_ROOT = SOURCES_ROOT / "raw"
RENDERED_ROOT = SOURCES_ROOT / "rendered"
PARSED_ROOT = SOURCES_ROOT / "parsed"
MANAGED_PREFIX = "scientific:"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(item: dict) -> Path:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    target = RAW_ROOT / f"{item['source_id']}.pdf"
    if target.exists() and target.stat().st_size > 10_000:
        return target
    request = Request(item["url"], headers={"User-Agent": "MOOSE-Research-Copilot/1.0 public-research-ingest"})
    with urlopen(request, timeout=45) as response:
        content_type = response.headers.get_content_type()
        payload = response.read()
    if content_type != "application/pdf" or not payload.startswith(b"%PDF"):
        raise ValueError(f"{item['source_id']} is not a PDF (content-type={content_type})")
    target.write_bytes(payload)
    return target


def _ocr_page(page: fitz.Page, source_id: str, number: int) -> tuple[str, float | None, str | None]:
    """Render a page and use the optional local OCR engine, preserving confidence."""
    RENDERED_ROOT.mkdir(parents=True, exist_ok=True)
    image_path = RENDERED_ROOT / f"{source_id}_p{number:03d}.png"
    page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(image_path)
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return "", None, str(image_path)
    engine = RapidOCR()
    result, _elapsed = engine(str(image_path))
    if not result:
        return "", None, str(image_path)
    texts = [str(row[1]) for row in result if len(row) >= 3 and str(row[1]).strip()]
    confidences = [float(row[2]) for row in result if len(row) >= 3]
    return "\n".join(texts), (mean(confidences) if confidences else None), str(image_path)


def parse_manifest(manifest_path: Path = MANIFEST_PATH) -> list[dict]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _block_documents(item: dict, pdf_path: Path, digest: str, ir: dict, ocr_map: dict[int, tuple[str, float | None, str | None]]) -> list[dict]:
    """Group nearby paragraphs while preserving table blocks and page-level parents."""
    output: list[dict] = []
    for page in ir["pages"]:
        pending: list[dict] = []
        pending_size = 0
        group_index = 0

        def flush() -> None:
            nonlocal pending, pending_size, group_index
            if not pending:
                return
            text = "\n\n".join(block["text"] for block in pending)
            page_number = page["page"]
            image = ocr_map.get(page_number, ("", None, None))[2]
            output.append({
                "document_id": f"{MANAGED_PREFIX}{item['source_id']}:page:{page_number}:group:{group_index}", "source_type": item["source_type"],
                "source_path": str(pdf_path), "content": text,
                "metadata": {"source_id": item["source_id"], "title": item["title"], "page": page_number, "parent_id": f"{item['source_id']}:page:{page_number}",
                             "authority": item["authority"], "access": item["access"], "url": item["url"], "topics": item["topics"], "aliases": item.get("aliases", []), "document_kind": item.get("document_kind", item["source_type"]),
                             "sha256": digest, "parser": "document_ir", "layout_parser": ir["parser"], "column_count": page["column_count"],
                             "block_type": "paragraph_group", "block_ids": [block["block_id"] for block in pending], "bboxes": [block["bbox"] for block in pending],
                             "section": pending[-1]["section"], "ocr_used": any(block["parser"] == "rapidocr" for block in pending),
                             "ocr_confidence": next((block["confidence"] for block in pending if block["confidence"] is not None), None), "page_image": image},
            })
            group_index += 1; pending = []; pending_size = 0

        for block in page["blocks"]:
            if block["block_type"] == "table":
                flush()
                output.append({
                    "document_id": f"{MANAGED_PREFIX}{item['source_id']}:page:{page['page']}:table:{block['block_id'].rsplit(':', 1)[-1]}", "source_type": item["source_type"],
                    "source_path": str(pdf_path), "content": block["text"],
                    "metadata": {"source_id": item["source_id"], "title": item["title"], "page": page["page"], "parent_id": f"{item['source_id']}:page:{page['page']}",
                                 "authority": item["authority"], "access": item["access"], "url": item["url"], "topics": item["topics"], "aliases": item.get("aliases", []), "document_kind": item.get("document_kind", item["source_type"]),
                                 "sha256": digest, "parser": block["parser"], "layout_parser": ir["parser"], "column_count": page["column_count"], "block_type": "table",
                                 "block_ids": [block["block_id"]], "bboxes": [block["bbox"]], "section": block["section"], "table_csv": block["table_csv"], "table_confidence": block["table_confidence"],
                                 "ocr_used": False, "ocr_confidence": None, "page_image": None},
                })
                continue
            if pending and pending_size + len(block["text"]) > 1800:
                flush()
            pending.append(block); pending_size += len(block["text"])
        flush()
    return output


def ingest_scientific_sources(*, download: bool = True, max_ocr_pages: int = 6, progress=None) -> dict:
    """Replace managed documents while isolating individual public-source failures."""
    items = parse_manifest()
    parsed: list[dict] = []
    source_summary: list[dict] = []
    failures: list[dict] = []
    for position, item in enumerate(items, 1):
        stage = "download"
        pdf = None
        try:
            if progress: progress("download", {"source_id": item["source_id"], "completed_sources": position - 1, "total_sources": len(items)})
            pdf_path = _download(item) if download else RAW_ROOT / f"{item['source_id']}.pdf"
            stage = "checksum"; digest = _sha256(pdf_path)
            stage = "open_pdf"; pdf = fitz.open(pdf_path)
            ocr_pages = 0
            confidences: list[float] = []
            ocr_map: dict[int, tuple[str, float | None, str | None]] = {}
            for page_index, page in enumerate(pdf, 1):
                extracted = page.get_text("text").strip()
                alpha_numeric = sum(char.isalnum() for char in extracted)
                # OCR is bounded and can be disabled for a large bulk ingest;
                # source-level errors must not discard all other public PDFs.
                needs_ocr = bool(item.get("force_ocr")) or len(extracted) < 80 or alpha_numeric < 60
                if needs_ocr and ocr_pages < max_ocr_pages:
                    ocr_text, confidence, image_path = _ocr_page(page, item["source_id"], page_index)
                    if ocr_text:
                        ocr_map[page_index] = (ocr_text, confidence, image_path)
                        ocr_pages += 1
                        if confidence is not None:
                            confidences.append(confidence)
            stage = "document_ir"
            if progress: progress(stage, {"source_id": item["source_id"], "completed_sources": position - 1, "total_sources": len(items)})
            ir = parse_pdf(pdf_path, item["source_id"], ocr_pages=ocr_map)
            stage = "document_groups"; source_documents = _block_documents(item, pdf_path, digest, ir, ocr_map)
            parsed.extend(source_documents)
            PARSED_ROOT.mkdir(parents=True, exist_ok=True)
            (PARSED_ROOT / f"{item['source_id']}.document_ir.json").write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")
            source_summary.append({"source_id": item["source_id"], "title": item["title"], "pages": len(pdf), "pages_ingested": len({document["metadata"]["page"] for document in source_documents}),
                                   "blocks_ingested": len(source_documents), "table_blocks": sum(document["metadata"]["block_type"] == "table" for document in source_documents),
                                   "two_column_pages": sum(page["column_count"] == 2 for page in ir["pages"]), "ocr_pages": ocr_pages,
                                   "mean_ocr_confidence": round(mean(confidences), 4) if confidences else None, "sha256": digest})
        except Exception as exc:
            failures.append({"source_id": item["source_id"], "stage": stage, "error_type": type(exc).__name__, "message": str(exc)[:300]})
            if progress: progress("source_failed", {"source_id": item["source_id"], "failure_count": len(failures), "completed_sources": position, "total_sources": len(items)})
        finally:
            if pdf is not None:
                pdf.close()
        if progress: progress("source_complete", {"source_id": item["source_id"], "completed_sources": position, "total_sources": len(items), "failure_count": len(failures)})
    if progress: progress("registry_replace", {"documents": len(parsed), "failures": len(failures), "total_sources": len(items)})
    SimulationRegistry(DB_PATH).replace_documents(MANAGED_PREFIX, parsed)
    PARSED_ROOT.mkdir(parents=True, exist_ok=True)
    result = {"sources": source_summary, "failures": failures, "documents_added": len(parsed), "ocr_engine_available": any(row["ocr_pages"] for row in source_summary)}
    (PARSED_ROOT / "ingest_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress: progress("completed", {"documents": len(parsed), "failures": len(failures), "total_sources": len(items)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--max-ocr-pages", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(ingest_scientific_sources(download=not args.no_download, max_ocr_pages=args.max_ocr_pages), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
