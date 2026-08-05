"""Constrained, observable LLM routing. Models may express evidence, never invent it."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app import config
from app.resource_limits import acquire_inference

ModelT = TypeVar("ModelT", bound=BaseModel)

@dataclass(frozen=True)
class ModelCall:
    role: str
    model: str
    latency_ms: float
    success: bool
    error: str = ""

class LLMRouter:
    """OpenAI-compatible router with role-specific local and API fallback chains."""
    def __init__(self, *, enabled: bool | None = None, client_factory=OpenAI) -> None:
        self.enabled = config.LLM_ENABLED if enabled is None else enabled
        self.client_factory = client_factory
        self.telemetry: list[ModelCall] = []

    def candidates(self, role: str) -> list[tuple[str, str, str]]:
        local = (config.LLM_BASE_URL, config.LLM_API_KEY)
        deepseek = (config.DEEPSEEK_BASE_URL, config.DEEPSEEK_API_KEY)
        chains = {
            "router": [(config.LLM_FAST_MODEL, *local), (config.LLM_PRIMARY_MODEL, *local)],
            "research": [(config.LLM_PRIMARY_MODEL, *local), (config.LLM_FAST_MODEL, *local)],
            "planner": [(config.LLM_PRIMARY_MODEL, *local), (config.LLM_FAST_MODEL, *local)],
            "critic": [(config.LLM_PRIMARY_MODEL, *local), (config.LLM_FAST_MODEL, *local)],
            "evidence": [(config.LLM_PRIMARY_MODEL, *local), (config.LLM_FAST_MODEL, *local)],
            "narrator": [(config.LLM_NARRATIVE_MODEL, *local), (config.LLM_PRIMARY_MODEL, *local)],
        }
        output = [item for item in chains[role] if item[0]]
        if config.DEEPSEEK_MODEL and all(deepseek): output.append((config.DEEPSEEK_MODEL, *deepseek))
        return output

    @staticmethod
    def _json(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text)

    def call_json(self, role: str, system: str, user: str, response_model: type[ModelT]) -> ModelT | None:
        if not self.enabled: return None
        schema = json.dumps(response_model.model_json_schema(), ensure_ascii=False)
        guarded_system = f"{system}\n仅输出一个符合此 JSON Schema 的对象，不得输出 Markdown：{schema}"
        for model, base_url, api_key in self.candidates(role):
            start = time.perf_counter()
            try:
                client = self.client_factory(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=0)
                # The local server may use CUDA; serializing only in-process
                # callers cannot control external Ollama, but prevents this
                # application from scheduling BGE work alongside it.
                local_gpu = base_url == config.LLM_BASE_URL and config.LLM_LOCAL_USES_GPU
                with acquire_inference("llm", uses_gpu=local_gpu):
                    response = client.chat.completions.create(model=model, temperature=0, messages=[{"role": "system", "content": guarded_system}, {"role": "user", "content": user}], response_format={"type": "json_object"})
                content = response.choices[0].message.content or ""
                parsed = response_model.model_validate(self._json(content))
                self.telemetry.append(ModelCall(role, model, round((time.perf_counter() - start) * 1000, 3), True))
                return parsed
            except Exception as exc:
                self.telemetry.append(ModelCall(role, model, round((time.perf_counter() - start) * 1000, 3), False, f"{type(exc).__name__}: {exc}"))
        return None
