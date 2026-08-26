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
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

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
            validation_hint = ""
            for attempt in range(2):
                start = time.perf_counter()
                try:
                    attempt_system = guarded_system if attempt == 0 else guarded_system + "\n上一次输出未通过协议校验。只修正 JSON 结构和字段类型，重新输出完整对象。"
                    attempt_user = user if attempt == 0 else user + f"\n请只修复上一次输出的协议错误，保持原始任务语义。错误详情：{validation_hint}\n重新输出完整 JSON 对象。"
                    client = self.client_factory(base_url=base_url, api_key=api_key, timeout=30.0, max_retries=0)
                    local_gpu = base_url == config.LLM_BASE_URL and config.LLM_LOCAL_USES_GPU
                    with acquire_inference("llm", uses_gpu=local_gpu):
                        response = client.chat.completions.create(model=model, temperature=0, messages=[{"role": "system", "content": attempt_system}, {"role": "user", "content": attempt_user}], response_format={"type": "json_object"})
                    content = response.choices[0].message.content or ""
                    usage = getattr(response, "usage", None)
                    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
                    cost_usd = input_tokens * float(config.LLM_INPUT_COST_PER_1M) / 1_000_000 + output_tokens * float(config.LLM_OUTPUT_COST_PER_1M) / 1_000_000
                    parsed = response_model.model_validate(self._json(content))
                    self.telemetry.append(ModelCall(role, model, round((time.perf_counter() - start) * 1000, 3), True, "", input_tokens, output_tokens, total_tokens, round(cost_usd, 8)))
                    return parsed
                except Exception as exc:
                    validation_hint = f"{type(exc).__name__}: {exc}"
                    self.telemetry.append(ModelCall(role, model, round((time.perf_counter() - start) * 1000, 3), False, f"{type(exc).__name__}: {exc}"))
        return None
