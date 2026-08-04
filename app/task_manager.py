"""Persistent, bounded background execution for research requests."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Callable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ResearchTaskStore:
    """Stores task state so a browser refresh does not lose observability."""
    def __init__(self, path: Path, max_workers: int = 4) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="research")
        self.futures: dict[str, Future] = {}
        self.lock = Lock()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS research_tasks (
                task_id TEXT PRIMARY KEY, question TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, started_at TEXT NOT NULL DEFAULT '', finished_at TEXT NOT NULL DEFAULT '',
                cancel_requested INTEGER NOT NULL DEFAULT 0, latency_ms REAL, result_json TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT ''
            )""")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10); db.row_factory = sqlite3.Row; return db

    def _row(self, row: sqlite3.Row) -> dict:
        return {**dict(row), "cancel_requested": bool(row["cancel_requested"]), "result": json.loads(row["result_json"])}

    def get(self, task_id: str) -> dict | None:
        with self._connect() as db: row = db.execute("SELECT * FROM research_tasks WHERE task_id=?", (task_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, limit: int = 50) -> list[dict]:
        with self._connect() as db: rows = db.execute("SELECT * FROM research_tasks ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(row) for row in rows]

    def submit(self, question: str, work: Callable[[], dict]) -> dict:
        task_id = "research-" + uuid.uuid4().hex[:12]
        with self._connect() as db:
            db.execute("INSERT INTO research_tasks(task_id,question,status,created_at) VALUES (?,?,?,?)", (task_id, question, "queued", _now()))
        future = self.executor.submit(self._run, task_id, work)
        with self.lock: self.futures[task_id] = future
        return self.get(task_id) or {}

    def _run(self, task_id: str, work: Callable[[], dict]) -> None:
        with self._connect() as db:
            row = db.execute("SELECT cancel_requested FROM research_tasks WHERE task_id=?", (task_id,)).fetchone()
            if row and row[0]:
                db.execute("UPDATE research_tasks SET status='cancelled',finished_at=? WHERE task_id=?", (_now(), task_id)); return
            db.execute("UPDATE research_tasks SET status='running',started_at=? WHERE task_id=?", (_now(), task_id))
        started = time.perf_counter()
        try:
            result = work()
            with self._connect() as db:
                cancelled = db.execute("SELECT cancel_requested FROM research_tasks WHERE task_id=?", (task_id,)).fetchone()[0]
                status = "cancelled_after_completion" if cancelled else "completed"
                db.execute("UPDATE research_tasks SET status=?,finished_at=?,latency_ms=?,result_json=? WHERE task_id=?", (status, _now(), round((time.perf_counter() - started) * 1000, 3), json.dumps(result, ensure_ascii=False), task_id))
        except Exception as exc:
            with self._connect() as db:
                db.execute("UPDATE research_tasks SET status='failed',finished_at=?,latency_ms=?,error=? WHERE task_id=?", (_now(), round((time.perf_counter() - started) * 1000, 3), f"{type(exc).__name__}: {exc}", task_id))

    def cancel(self, task_id: str) -> dict:
        task = self.get(task_id)
        if not task: raise KeyError(task_id)
        if task["status"] in {"completed", "failed", "cancelled", "cancelled_after_completion"}: return task
        with self._connect() as db: db.execute("UPDATE research_tasks SET cancel_requested=1,status='cancellation_requested' WHERE task_id=?", (task_id,))
        with self.lock:
            future = self.futures.get(task_id)
            if future and future.cancel():
                with self._connect() as db: db.execute("UPDATE research_tasks SET status='cancelled',finished_at=? WHERE task_id=?", (_now(), task_id))
        return self.get(task_id) or {}

    def metrics(self) -> dict:
        rows = self.list(500); completed = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
        return {"total": len(rows), "by_status": {status: sum(row["status"] == status for row in rows) for status in sorted({row["status"] for row in rows})}, "mean_latency_ms": round(sum(completed) / len(completed), 3) if completed else None, "max_workers": self.executor._max_workers}
