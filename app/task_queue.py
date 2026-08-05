"""Optional Redis/RQ task dispatch with a local bounded fallback."""
from __future__ import annotations

from collections.abc import Callable

from app import config
from app.task_manager import ResearchTaskStore


class ResearchTaskDispatcher:
    def __init__(self, store: ResearchTaskStore, work: Callable[[str, str], None]) -> None:
        self.store = store
        self.work = work
        self.backend, self.queue = self._queue()

    def _queue(self):
        if config.TASK_BACKEND == "local": return "local", None
        try:
            from redis import Redis
            from rq import Queue
            # RQ persists pickled job metadata; decoding Redis values as UTF-8
            # corrupts that binary payload when a worker fetches the job.
            connection = Redis.from_url(config.REDIS_URL, socket_connect_timeout=1, protocol=2)
            connection.ping()
            return "rq", Queue(config.TASK_QUEUE_NAME, connection=connection, default_timeout=900)
        except Exception as exc:
            if config.TASK_BACKEND == "rq": raise RuntimeError("TASK_BACKEND=rq but Redis/RQ is unavailable") from exc
            return "local", None

    def submit(self, question: str) -> dict:
        task = self.store.create(question, queue_backend=self.backend)
        if self.backend == "rq":
            self.queue.enqueue(self.work, task["task_id"], question, job_id=task["task_id"], result_ttl=86400, failure_ttl=86400)
        else:
            future = self.store.executor.submit(self.work, task["task_id"], question)
            with self.store.lock:
                self.store.futures[task["task_id"]] = future
        return task

    def metrics(self) -> dict:
        return {"configured_backend": config.TASK_BACKEND, "active_backend": self.backend, "queue_name": config.TASK_QUEUE_NAME, **self.store.metrics()}
