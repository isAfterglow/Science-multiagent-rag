"""Dependency-free request tracing shared by the graph, task API and evals."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "agent_trace.v1"
PROJECT = "moose-research-copilot"


def new_trace_id() -> str:
    return "research-" + uuid.uuid4().hex[:16]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def normalize_event(event: dict[str, Any], *, trace_id: str, parent_span_id: str, elapsed_ms: float) -> dict[str, Any]:
    """Preserve agent-specific fields while adding the common Trace contract."""
    node = str(event.get("node", "unknown"))
    status = "rejected" if node == "reviewer" and not event.get("approved", True) else "ok"
    return {
        **event,
        "schema_version": SCHEMA_VERSION,
        "project": PROJECT,
        "trace_id": trace_id,
        "span_id": uuid.uuid4().hex[:16],
        "parent_span_id": parent_span_id,
        "event_type": f"agent.{status}",
        "status": status,
        "finished_at": now(),
        "elapsed_ms": round(elapsed_ms, 3),
    }


def summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "node_path": [str(event.get("node", "")) for event in events],
        "node_count": len(events),
        "rejected_nodes": [str(event.get("node", "")) for event in events if event.get("status") == "rejected"],
        "total_node_elapsed_ms": round(sum(float(event.get("elapsed_ms", 0.0)) for event in events), 3),
    }
