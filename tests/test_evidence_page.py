import pytest
from fastapi import HTTPException

from app import api


def test_managed_evidence_pdf_rejects_invalid_source_id(monkeypatch):
    monkeypatch.setattr(api.registry, "documents", lambda: [])
    with pytest.raises(HTTPException) as exc:
        api._managed_evidence_pdf("../../etc/passwd")
    assert exc.value.status_code == 404


def test_managed_evidence_pdf_rejects_unmanaged_pdf(monkeypatch, tmp_path):
    outside_pdf = tmp_path / "outside.pdf"
    monkeypatch.setattr(api.registry, "documents", lambda: [{"source_path": str(outside_pdf), "metadata": {"source_id": "public_doc"}}])
    with pytest.raises(HTTPException) as exc:
        api._managed_evidence_pdf("public_doc")
    assert exc.value.status_code == 403


def test_managed_evidence_pdf_accepts_registered_raw_pdf(monkeypatch):
    raw_pdf = (api.ROOT / "data" / "knowledge_sources" / "raw" / "public_doc.pdf").resolve()
    monkeypatch.setattr(api.registry, "documents", lambda: [{"source_path": str(raw_pdf), "metadata": {"source_id": "public_doc"}}])
    assert api._managed_evidence_pdf("public_doc") == raw_pdf
