from __future__ import annotations

from app.resource_limits import acquire_inference, metrics


def test_gpu_inference_admission_records_metrics():
    before = metrics()["gpu_inference"]["requests"]
    with acquire_inference("embedding", uses_gpu=True) as waits:
        assert waits["model_wait_ms"] >= 0
        assert waits["gpu_wait_ms"] >= 0
    assert metrics()["gpu_inference"]["requests"] == before + 1
