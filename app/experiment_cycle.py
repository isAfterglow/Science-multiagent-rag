"""Turn a documented knowledge gap into an approval-gated experiment cycle."""
from __future__ import annotations

import re
from typing import Any

from app.analysis import parameter_correlation
from app.registry import SimulationRegistry
from app.simulation_plan import PARAMETER_BOUNDS, PlanStore, suggest_exploration_plan


def diagnose_gap(question: str, registry: SimulationRegistry, metric: str = "early_1_2_rmse") -> dict[str, Any]:
    mentioned = sorted(set(re.findall(r"[a-z]+_[a-z]+_scale|tbegin\d+_shift", question.lower())))
    known = [name for name in mentioned if name in PARAMETER_BOUNDS]
    unknown = [name for name in mentioned if name not in PARAMETER_BOUNDS]
    # A generic request for a simulation recommendation is not itself a
    # knowledge gap. It is handled by the separate exploration-plan path.
    intent = any(token in question for token in ("未记录", "没有数据", "新工况", "需要验证", "验证", "实验"))
    if unknown:
        return {"status": "unsupported", "gap_type": "unknown_parameter", "unknown_parameters": unknown, "message": "参数不在仿真白名单中，不能自动设计实验。"}
    if not intent:
        return {"status": "not_required", "gap_type": "retrieval_or_analysis", "message": "问题未明确要求新实验；应优先使用历史证据或拒绝无证据结论。"}
    correlation = parameter_correlation(registry, metric)["correlations"]
    variables = known or sorted(correlation, key=lambda name: abs(correlation[name]), reverse=True)[:3]
    return {"status": "needs_experiment", "gap_type": "missing_condition_evidence", "target_metric": metric, "variable_parameters": variables, "baseline_case": registry.top_cases(metric, 1)[0]["case_id"], "success_criteria": f"在登记模板和参数边界内，对比基线并报告 {metric}；不能将单次结果解释为因果证明。", "message": "历史 Registry 不包含该新条件的直接证据，需要受控新仿真。"}


def design_experiment(question: str, registry: SimulationRegistry, metric: str = "early_1_2_rmse") -> dict[str, Any]:
    gap = diagnose_gap(question, registry, metric)
    if gap["status"] != "needs_experiment":
        return {"gap": gap, "plan": None}
    count = min(3, max(1, len(gap["variable_parameters"])))
    plan = suggest_exploration_plan(registry, metric, count)
    return {"gap": gap, "plan": plan.model_dump(), "execution_policy": "draft_only_requires_explicit_confirmation_and_human_approval"}


def confirm_experiment_draft(plan_payload: dict[str, Any], store: PlanStore) -> dict[str, Any]:
    """Explicitly persist a validated draft; it remains pending for review."""
    from app.models import SimulationPlan
    return store.create(SimulationPlan.model_validate(plan_payload))


def validate_and_report_result(plan_id: str, store: PlanStore, registry: SimulationRegistry) -> dict[str, Any]:
    record = store.get(plan_id)
    if not record:
        raise KeyError(plan_id)
    result = record["result"]
    if not result:
        return {"plan_id": plan_id, "status": "not_executed", "report": "计划尚未执行，不能生成科学结论。"}
    statuses = [item["status"] for item in result.get("results", [])]
    if result.get("dry_run"):
        conclusion = "仅完成隔离 dry-run：输入、参数和执行清单已验证，未产生新的 MOOSE 科学指标。"
    elif any(status != "ok" for status in statuses):
        conclusion = "执行未全部成功，已保留失败分类和日志；不将失败或环境阻断结果写成科学结论。"
    else:
        conclusion = "执行成功；新 case 已进入 Registry。应将新指标与基线和实验数据复核后再报告因果结论。"
    return {"plan_id": plan_id, "status": record["status"], "result_statuses": statuses, "registry_case_ids": [f"{plan_id}-{item['label']}" for item in result.get("results", [])], "report": conclusion}
