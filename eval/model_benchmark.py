"""Live local-model selection benchmark for constrained JSON tasks."""
from __future__ import annotations

import json
import time
from statistics import mean

from openai import OpenAI

from app import config
from app.llm_protocol import EvidenceSummary, PlanHint

CASES = [
    ("planner", PlanHint, "将问题转为检索计划：cpv_front_scale 对 early_1_2_rmse 有何影响？", "early_1_2_rmse"),
    ("planner", PlanHint, "将问题转为检索计划：如何解释 temp_mean_rmse 的历史基线？", "temp_mean_rmse"),
    ("evidence", EvidenceSummary, "仅根据此摘录整理，不新增数值：报告指出 cpv_front_scale 与 early_1_2_rmse 存在负相关。", None),
]
ALLOWED_METRICS = "early_1_2_rmse, temp_mean_rmse"

def _call(client: OpenAI, model: str, role: str, schema: type, question: str):
    contract = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    system = f"你是 {role}。返回用于本次任务的 JSON 实例，绝不能返回 Schema、properties 或 explanation。仅输出一个 JSON 对象，不得使用 Markdown。若存在 analysis_metric，它必须严格从 [{ALLOWED_METRICS}] 选择并直接复制问题中出现的指标。Schema：{contract}"
    response = client.chat.completions.create(model=model, temperature=0, max_tokens=220, response_format={"type": "json_object"}, messages=[{"role": "system", "content": system}, {"role": "user", "content": question}])
    raw = response.choices[0].message.content or ""
    return schema.model_validate_json(raw), raw

def run(models: list[str] | None = None) -> dict:
    models = models or [config.LLM_FAST_MODEL, config.LLM_PRIMARY_MODEL, config.LLM_NARRATIVE_MODEL]
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, timeout=45.0, max_retries=0)
    results = []
    for model in dict.fromkeys(models):
        case_rows = []
        for role, schema, question, expected_metric in CASES:
            started = time.perf_counter()
            try:
                output, raw = _call(client, model, role, schema, question)
                metric_ok = expected_metric is None or getattr(output, "analysis_metric", "") == expected_metric
                case_rows.append({"role": role, "success": True, "schema_valid": True, "task_valid": metric_ok, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "raw": raw})
            except Exception as exc:
                case_rows.append({"role": role, "success": False, "schema_valid": False, "task_valid": False, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "error": f"{type(exc).__name__}: {exc}"})
        valid = sum(row["schema_valid"] and row["task_valid"] for row in case_rows)
        results.append({"model": model, "valid": valid, "total": len(case_rows), "valid_rate": round(valid / len(case_rows), 4), "mean_latency_ms": round(mean(row["latency_ms"] for row in case_rows), 3), "cases": case_rows})
    available = [item for item in results if item["valid"]]
    winner = sorted(available, key=lambda item: (-item["valid_rate"], item["mean_latency_ms"]))[0]["model"] if available else None
    return {"base_url": config.LLM_BASE_URL, "models": results, "winner": winner, "selection_rule": "highest valid JSON/task rate, then lowest mean latency"}

if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
