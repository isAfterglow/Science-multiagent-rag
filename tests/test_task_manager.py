import time

from app.task_manager import ResearchTaskStore


def test_persistent_background_task_lifecycle(tmp_path):
    store = ResearchTaskStore(tmp_path / "tasks.sqlite3", max_workers=1)
    task = store.submit("test", lambda: {"answer": "ok", "trace": []})
    deadline = time.time() + 3
    while time.time() < deadline:
        current = store.get(task["task_id"])
        if current and current["status"] == "completed":
            break
        time.sleep(0.01)
    assert current["status"] == "completed"
    assert current["result"]["answer"] == "ok"
    assert store.metrics()["total"] == 1
