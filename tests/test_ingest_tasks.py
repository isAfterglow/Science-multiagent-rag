from __future__ import annotations

from app.ingest_tasks import ScientificIngestTasks


def test_ingest_task_persists_submission(tmp_path, monkeypatch):
    monkeypatch.setattr("app.ingest_tasks.ingest_scientific_sources", lambda **_kwargs: {"documents_added": 0, "failures": []})
    tasks = ScientificIngestTasks(tmp_path / "tasks.sqlite3")
    task = tasks.submit(download=False, max_ocr_pages=0)
    stored = tasks.get(task["task_id"])
    assert stored and stored["status"] in {"queued", "running", "completed", "failed"}
    tasks.shutdown()
