import json
from types import SimpleNamespace

from app.llm_protocol import RouteDecision
from app.llm_router import LLMRouter


class _Response:
    def __init__(self, content):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)


class _Client:
    outputs = []
    def __init__(self, **_kwargs): pass
    class chat:
        class completions:
            @staticmethod
            def create(**_kwargs): return _Response(_Client.outputs.pop(0))


def test_validation_guided_repair_recovers_invalid_enum(monkeypatch):
    _Client.outputs = [json.dumps({"task_type": "analysis"}), json.dumps({"task_type": "knowledge", "needs_registry_analysis": False, "required_sources": []})]
    router = LLMRouter(enabled=True, client_factory=_Client)
    result = router.call_json("router", "route", "question", RouteDecision)
    assert result and result.task_type == "knowledge"
    assert len(router.telemetry) == 2
    assert router.telemetry[0].success is False and router.telemetry[1].success is True


def test_validation_guided_repair_rejects_schema_object(monkeypatch):
    _Client.outputs = [json.dumps({"properties": {"task_type": {"type": "string"}}}), json.dumps({"task_type": "knowledge", "needs_registry_analysis": False, "required_sources": []})]
    router = LLMRouter(enabled=True, client_factory=_Client)
    result = router.call_json("router", "route", "question", RouteDecision)
    assert result and result.task_type == "knowledge"
    assert "ValidationError" in router.telemetry[0].error
