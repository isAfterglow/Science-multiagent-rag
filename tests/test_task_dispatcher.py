import time

import pytest

from app import config
from app.config import SOURCE_DIR
import app.research_worker as research_worker
from app.task_manager import ResearchTaskStore
from app.task_queue import ResearchTaskDispatcher


def test_local_dispatcher_persists_backend_and_queue_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TASK_BACKEND", "local")
    store = ResearchTaskStore(tmp_path / "tasks.sqlite3", max_workers=1)

    def work(task_id: str, _question: str) -> None:
        assert store.claim(task_id)
        started = time.perf_counter()
        store.complete(task_id, {"answer": "ok", "trace": []}, started)

    dispatcher = ResearchTaskDispatcher(store, work)
    task = dispatcher.submit("test")
    deadline = time.time() + 3
    while time.time() < deadline:
        current = store.get(task["task_id"])
        if current and current["status"] == "completed": break
        time.sleep(0.01)
    assert current["queue_backend"] == "local"
    assert current["queue_wait_ms"] is not None
    assert dispatcher.metrics()["active_backend"] == "local"


def test_worker_adds_queue_event_to_task_trace(tmp_path, monkeypatch):
    task_db, registry_db = tmp_path / "tasks.sqlite3", tmp_path / "registry.sqlite3"
    monkeypatch.setattr(research_worker, "TASK_DB_PATH", task_db)
    monkeypatch.setattr(research_worker, "DB_PATH", registry_db)
    monkeypatch.setattr(research_worker, "SOURCE_DIR", SOURCE_DIR)
    store = ResearchTaskStore(task_db, max_workers=1)
    task = store.create("early_1_2_rmse 最好的 case 是哪个", queue_backend="local")
    research_worker.run_research_task(task["task_id"], task["question"])
    completed = store.get(task["task_id"])
    assert completed["status"] == "completed"
    first_event = completed["result"]["trace"][0]
    assert first_event["node"] == "task_dispatch"
    assert first_event["queue_backend"] == "local"
    assert first_event["trace_id"] == completed["result"]["trace_id"]


def test_auto_falls_back_but_forced_rq_fails_closed(tmp_path, monkeypatch):
    store = ResearchTaskStore(tmp_path / "tasks.sqlite3", max_workers=1)
    monkeypatch.setattr(config, "REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setattr(config, "TASK_BACKEND", "auto")
    assert ResearchTaskDispatcher(store, lambda *_: None).backend == "local"
    monkeypatch.setattr(config, "TASK_BACKEND", "rq")
    with pytest.raises(RuntimeError, match="Redis/RQ"):
        ResearchTaskDispatcher(store, lambda *_: None)
