"""Persistent local scientific-ingest jobs with stage-level observability."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.scientific_ingest import ingest_scientific_sources


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ScientificIngestTasks:
    """Keep download/parse/OCR/index work off the HTTP request thread."""
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.executor = path, ThreadPoolExecutor(max_workers=1, thread_name_prefix="scientific-ingest")
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS scientific_ingest_tasks (
                task_id TEXT PRIMARY KEY, status TEXT NOT NULL, stage TEXT NOT NULL,
                created_at TEXT NOT NULL, started_at TEXT NOT NULL DEFAULT '', finished_at TEXT NOT NULL DEFAULT '',
                progress_json TEXT NOT NULL DEFAULT '{}', result_json TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT '', elapsed_ms REAL
            )""")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=15); db.row_factory = sqlite3.Row; return db

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["progress"] = json.loads(value.pop("progress_json"))
        value["result"] = json.loads(value.pop("result_json"))
        return value

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as db: row = db.execute("SELECT * FROM scientific_ingest_tasks WHERE task_id=?", (task_id,)).fetchone()
        return self._row(row) if row else None

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as db: rows = db.execute("SELECT * FROM scientific_ingest_tasks ORDER BY created_at DESC LIMIT 30").fetchall()
        return [self._row(row) for row in rows]

    def submit(self, *, download: bool = True, max_ocr_pages: int = 6) -> dict[str, Any]:
        task_id = "ingest-" + uuid.uuid4().hex[:12]
        with self._connect() as db:
            db.execute("INSERT INTO scientific_ingest_tasks(task_id,status,stage,created_at,progress_json) VALUES (?,?,?,?,?)", (task_id, "queued", "queued", _now(), json.dumps({"download": download, "max_ocr_pages": max_ocr_pages})))
        self.executor.submit(self._run, task_id, download, max_ocr_pages)
        return self.get(task_id) or {}

    def shutdown(self) -> None:
        """Release the local worker in tests or an orderly application stop."""
        self.executor.shutdown(wait=True)

    def _update(self, task_id: str, stage: str, payload: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("UPDATE scientific_ingest_tasks SET stage=?,progress_json=? WHERE task_id=?", (stage, json.dumps(payload, ensure_ascii=False), task_id))

    def _run(self, task_id: str, download: bool, max_ocr_pages: int) -> None:
        started = time.perf_counter()
        with self._connect() as db: db.execute("UPDATE scientific_ingest_tasks SET status='running',started_at=? WHERE task_id=?", (_now(), task_id))
        try:
            def progress(stage: str, payload: dict[str, Any]) -> None:
                self._update(task_id, stage, {**payload, "device": "cpu", "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)})
            result = ingest_scientific_sources(download=download, max_ocr_pages=max_ocr_pages, progress=progress)
            with self._connect() as db: db.execute("UPDATE scientific_ingest_tasks SET status='completed',stage='completed',finished_at=?,elapsed_ms=?,result_json=? WHERE task_id=?", (_now(), round((time.perf_counter() - started) * 1000, 3), json.dumps(result, ensure_ascii=False), task_id))
        except Exception as exc:
            with self._connect() as db: db.execute("UPDATE scientific_ingest_tasks SET status='failed',stage='failed',finished_at=?,elapsed_ms=?,error=? WHERE task_id=?", (_now(), round((time.perf_counter() - started) * 1000, 3), f"{type(exc).__name__}: {exc}", task_id))
