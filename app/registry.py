"""SQLite registry for immutable historical MOOSE simulation evidence."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class SimulationRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS simulation_cases (
              case_id TEXT PRIMARY KEY, parameters_json TEXT NOT NULL,
              status TEXT NOT NULL, return_code INTEGER, elapsed_s REAL,
              source_path TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metrics (
              case_id TEXT NOT NULL, metric_name TEXT NOT NULL, metric_value REAL NOT NULL,
              PRIMARY KEY(case_id, metric_name)
            );
            CREATE TABLE IF NOT EXISTS artifacts (
              case_id TEXT NOT NULL, artifact_type TEXT NOT NULL, path TEXT NOT NULL,
              PRIMARY KEY(case_id, artifact_type, path)
            );
            CREATE TABLE IF NOT EXISTS documents (
              document_id TEXT PRIMARY KEY, source_type TEXT NOT NULL, source_path TEXT NOT NULL,
              content TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_metrics_name_value ON metrics(metric_name, metric_value);
            CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(source_type);
            """)
            # Existing project registries predate scientific-source metadata.
            # Add it in-place so importing papers never resets case evidence.
            columns = {row["name"] for row in db.execute("PRAGMA table_info(documents)").fetchall()}
            if "metadata_json" not in columns:
                db.execute("ALTER TABLE documents ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")

    def reset(self) -> None:
        with self._connect() as db:
            for table in ("simulation_cases", "metrics", "artifacts", "documents"):
                db.execute(f"DELETE FROM {table}")

    def upsert_case(self, case_id: str, parameters: dict[str, float], status: str, return_code: int | None, elapsed_s: float | None, source_path: str) -> None:
        with self._connect() as db:
            db.execute("""INSERT INTO simulation_cases VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET parameters_json=excluded.parameters_json,
                status=excluded.status, return_code=excluded.return_code, elapsed_s=excluded.elapsed_s, source_path=excluded.source_path""",
                (case_id, json.dumps(parameters, sort_keys=True), status, return_code, elapsed_s, source_path))

    def upsert_metric(self, case_id: str, name: str, value: float) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO metrics VALUES (?, ?, ?)", (case_id, name, value))

    def add_artifact(self, case_id: str, artifact_type: str, path: str) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?)", (case_id, artifact_type, path))

    def add_document(self, document_id: str, source_type: str, source_path: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO documents(document_id, source_type, source_path, content, metadata_json) VALUES (?, ?, ?, ?, ?)",
                       (document_id, source_type, source_path, content, json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)))

    def replace_documents(self, prefix: str, documents: list[dict[str, Any]]) -> None:
        """Atomically replace a managed document namespace without touching cases."""
        with self._connect() as db:
            db.execute("DELETE FROM documents WHERE document_id LIKE ?", (f"{prefix}%",))
            db.executemany(
                "INSERT INTO documents(document_id, source_type, source_path, content, metadata_json) VALUES (?, ?, ?, ?, ?)",
                [(item["document_id"], item["source_type"], item["source_path"], item["content"],
                  json.dumps(item.get("metadata", {}), ensure_ascii=False, sort_keys=True)) for item in documents],
            )

    def cases(self) -> list[dict[str, Any]]:
        with self._connect() as db: rows = db.execute("SELECT * FROM simulation_cases ORDER BY case_id").fetchall()
        return [{**dict(row), "parameters": json.loads(row["parameters_json"])} for row in rows]

    def top_cases(self, metric: str, limit: int = 5) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("""SELECT c.case_id, m.metric_value, c.parameters_json, c.status, c.elapsed_s
                FROM metrics m JOIN simulation_cases c USING(case_id) WHERE m.metric_name = ?
                ORDER BY m.metric_value ASC LIMIT ?""", (metric, limit)).fetchall()
        return [{**dict(row), "parameters": json.loads(row["parameters_json"])} for row in rows]

    def case_detail(self, case_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            case = db.execute("SELECT * FROM simulation_cases WHERE case_id = ?", (case_id,)).fetchone()
            if not case: return None
            metrics = db.execute("SELECT metric_name, metric_value FROM metrics WHERE case_id = ? ORDER BY metric_name", (case_id,)).fetchall()
            artifacts = db.execute("SELECT artifact_type, path FROM artifacts WHERE case_id = ?", (case_id,)).fetchall()
        return {**dict(case), "parameters": json.loads(case["parameters_json"]), "metrics": dict(metrics), "artifacts": [dict(row) for row in artifacts]}

    def metric_frame(self, metric: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("""SELECT c.parameters_json, m.case_id, m.metric_value FROM metrics m
                JOIN simulation_cases c USING(case_id) WHERE m.metric_name = ?""", (metric,)).fetchall()
        return [{"case_id": row["case_id"], metric: row["metric_value"], **json.loads(row["parameters_json"])} for row in rows]

    def documents(self) -> list[dict[str, Any]]:
        with self._connect() as db: rows = db.execute("SELECT * FROM documents ORDER BY document_id").fetchall()
        return [{**dict(row), "metadata": json.loads(row["metadata_json"] or "{}") } for row in rows]
