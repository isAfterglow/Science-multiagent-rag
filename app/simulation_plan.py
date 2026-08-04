"""Constrained simulation plans, approval persistence, and safe case construction."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

from app.config import RUN_SCRIPT_PATH, RUNS_DIR, SURROGATE_DIR, TEMPLATE_PATH
from app.models import ParameterSet, SimulationPlan
from app.registry import SimulationRegistry
from app.analysis import parameter_correlation

PARAMETER_BOUNDS = {
    "preexp_scale": (0.35, 3.00), "ER_scale": (0.82, 1.22), "tbegin1_shift": (-35.0, 70.0),
    "tbegin2_shift": (-50.0, 90.0), "pyrolysis_heat_scale": (0.70, 1.35), "cpv_front_scale": (0.95, 2.25),
    "cpv_mid_scale": (0.90, 1.70), "kv_front_scale": (0.80, 1.15), "kv_mid_scale": (0.85, 1.20),
    "rho_reactive_scale": (0.90, 1.18),
}

BASE = {
    "rhov_i": [300.0, 900.0, 1600.0], "rhoc_i": [0.0, 600.0, 1600.0], "T_begin_i": [333.0, 556.0, 5560.0],
    "precofficient_i": [12000.0, 4.48e9, 0.0], "ER_i": [8556.0, 2.044e4, 0.0],
    "kv_list": [0.3975, 0.4025, 0.4162, 0.4530, 0.4698, 0.4860, 0.5234, 0.5601, 0.6978, 0.8723, 1.1090, 1.7510, 2.7790],
    "cpv_list": [879.2, 983.9, 1298.0, 1465.0, 1570.0, 1717.0, 1863.0, 1934.0, 1980.0, 1989.0, 2001.0, 2010.0, 2010.0], "pyrolysis_heat": -8.571e5,
}

def _now() -> str: return datetime.now(timezone.utc).isoformat(timespec="seconds")
def template_hash() -> str: return hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()

def validate_plan(plan: SimulationPlan) -> list[str]:
    errors: list[str] = []
    if plan.template_sha256 != template_hash(): errors.append("输入模板哈希不匹配，拒绝执行未登记模板。")
    for case in plan.cases:
        if set(case.values) != set(PARAMETER_BOUNDS): errors.append(f"{case.label}: 参数集合必须与白名单完全一致。")
        for name, value in case.values.items():
            if name in PARAMETER_BOUNDS:
                low, high = PARAMETER_BOUNDS[name]
                if not low <= value <= high: errors.append(f"{case.label}: {name}={value} 超出允许范围 [{low}, {high}]。")
    return errors

class PlanStore:
    def __init__(self, path: Path) -> None:
        self.path = path; path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS simulation_plans (
                plan_id TEXT PRIMARY KEY, status TEXT NOT NULL, payload_json TEXT NOT NULL,
                validation_json TEXT NOT NULL, decision_actor TEXT NOT NULL DEFAULT '', decision_comment TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL, decided_at TEXT NOT NULL DEFAULT '', executed_at TEXT NOT NULL DEFAULT '', result_json TEXT NOT NULL DEFAULT '{}')""")
    def _connect(self):
        db = sqlite3.connect(self.path); db.row_factory = sqlite3.Row; return db
    def create(self, plan: SimulationPlan) -> dict[str, Any]:
        errors = validate_plan(plan)
        status = "pending" if not errors else "rejected_validation"
        with self._connect() as db: db.execute("INSERT INTO simulation_plans(plan_id,status,payload_json,validation_json,created_at) VALUES (?,?,?,?,?)", (plan.plan_id, status, plan.model_dump_json(), json.dumps(errors, ensure_ascii=False), _now()))
        return self.get(plan.plan_id) or {}
    def get(self, plan_id: str) -> dict[str, Any] | None:
        with self._connect() as db: row = db.execute("SELECT * FROM simulation_plans WHERE plan_id=?", (plan_id,)).fetchone()
        return self._row(row) if row else None
    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db: rows = db.execute("SELECT * FROM simulation_plans ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._row(row) for row in rows]
    def decide(self, plan_id: str, action: str, actor: str, comment: str = "") -> dict[str, Any]:
        if action not in {"approved", "rejected"}: raise ValueError("action must be approved or rejected")
        row = self.get(plan_id)
        if not row: raise KeyError(plan_id)
        if row["status"] != "pending": raise ValueError(f"plan is not pending: {row['status']}")
        with self._connect() as db: db.execute("UPDATE simulation_plans SET status=?,decision_actor=?,decision_comment=?,decided_at=? WHERE plan_id=?", (action, actor, comment, _now(), plan_id))
        return self.get(plan_id) or {}
    def save_result(self, plan_id: str, result: dict[str, Any]) -> None:
        with self._connect() as db: db.execute("UPDATE simulation_plans SET status='executed',executed_at=?,result_json=? WHERE plan_id=?", (_now(), json.dumps(result, ensure_ascii=False), plan_id))
    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        return {**dict(row), "plan": json.loads(row["payload_json"]), "validation_errors": json.loads(row["validation_json"]), "result": json.loads(row["result_json"])}

def _scale(values: list[float], front: float, mid: float) -> list[float]:
    return [value * front if i <= 5 else value * mid if i <= 10 else value for i, value in enumerate(values)]

def _replace_line(text: str, key: str, value: str) -> str:
    import re
    output, count = re.subn(rf"(^\s*{re.escape(key)}\s*=\s*)([^\n]+)$", rf"\g<1>{value}", text, flags=re.MULTILINE)
    if count != 1: raise ValueError(f"template field not found: {key}")
    return output

def build_input_deck(values: dict[str, float]) -> str:
    text = TEMPLATE_PATH.read_text(encoding="utf-8")
    payload = {
        "rhov_i": [BASE["rhov_i"][0] * values["rho_reactive_scale"], BASE["rhov_i"][1] * values["rho_reactive_scale"], BASE["rhov_i"][2]],
        "rhoc_i": [BASE["rhoc_i"][0] * values["rho_reactive_scale"], BASE["rhoc_i"][1] * values["rho_reactive_scale"], BASE["rhoc_i"][2]],
        "T_begin_i": [BASE["T_begin_i"][0] + values["tbegin1_shift"], BASE["T_begin_i"][1] + values["tbegin2_shift"], BASE["T_begin_i"][2]],
        "precofficient_i": [BASE["precofficient_i"][0] * values["preexp_scale"], BASE["precofficient_i"][1] * values["preexp_scale"], 0.0],
        "ER_i": [BASE["ER_i"][0] * values["ER_scale"], BASE["ER_i"][1] * values["ER_scale"], 0.0],
        "cpv_list": _scale(BASE["cpv_list"], values["cpv_front_scale"], values["cpv_mid_scale"]),
        "kv_list": _scale(BASE["kv_list"], values["kv_front_scale"], values["kv_mid_scale"]),
    }
    for key, items in payload.items(): text = _replace_line(text, key, "'" + " ".join(f"{item:.8g}" for item in items) + "'")
    return _replace_line(text, "pyrolysis_heat", f"{BASE['pyrolysis_heat'] * values['pyrolysis_heat_scale']:.8g}")

def suggest_plan(registry: SimulationRegistry, target_metric: str = "early_1_2_rmse", n_cases: int = 3) -> SimulationPlan:
    if not 1 <= n_cases <= 5: raise ValueError("n_cases must be between 1 and 5")
    top = registry.top_cases(target_metric, max(n_cases, 1)); elapsed = [row["elapsed_s"] for row in registry.cases() if row["elapsed_s"]]
    cases = [ParameterSet(label=f"candidate-{index + 1}", values=dict(row["parameters"]), surrogate_prediction=float(row["metric_value"])) for index, row in enumerate(top[:n_cases])]
    return SimulationPlan(plan_id="plan-" + uuid.uuid4().hex[:12], target_metric=target_metric, template_sha256=template_hash(), cases=cases, estimated_seconds=round((median(elapsed) if elapsed else 60.0) * n_cases, 2), max_case_timeout_seconds=180, rationale=f"基于历史 {target_metric} Top-{n_cases} case 的参数组合生成候选计划；代理预测采用历史实测指标，仅用于优先级排序。")


def suggest_exploration_plan(registry: SimulationRegistry, target_metric: str = "early_1_2_rmse", n_cases: int = 3) -> SimulationPlan:
    """Create bounded, differentiated candidates from correlation direction.

    This is a proposal, not a prediction or autonomous execution decision.
    """
    if not 1 <= n_cases <= 3:
        raise ValueError("exploration plan supports one to three candidates")
    baseline = registry.top_cases(target_metric, 1)
    if not baseline:
        raise ValueError(f"No historical cases for metric: {target_metric}")
    base = dict(baseline[0]["parameters"])
    correlations = parameter_correlation(registry, target_metric)["correlations"]
    selected = sorted(correlations, key=lambda name: abs(correlations[name]), reverse=True)[:n_cases]
    cases: list[ParameterSet] = []
    descriptions: list[str] = []
    for index, name in enumerate(selected, 1):
        values = dict(base); low, high = PARAMETER_BOUNDS[name]
        direction = -1 if correlations[name] > 0 else 1
        # Use 5% of the allowed span, preserving parameter-specific units.
        proposed = min(high, max(low, values[name] + direction * (high - low) * 0.05))
        values[name] = round(proposed, 8)
        cases.append(ParameterSet(label=f"explore-{index}-{name}", values=values))
        descriptions.append(f"{name} Pearson={correlations[name]:+.3f}，按降低 {target_metric} 的相关方向在允许区间内扰动 5% span")
    elapsed = [row["elapsed_s"] for row in registry.cases() if row["elapsed_s"]]
    return SimulationPlan(
        plan_id="draft-" + uuid.uuid4().hex[:12], target_metric=target_metric, template_sha256=template_hash(), cases=cases,
        estimated_seconds=round((median(elapsed) if elapsed else 60.0) * len(cases), 2), max_case_timeout_seconds=180,
        rationale="探索性候选以当前历史最优 case 为基线；" + "；".join(descriptions) + "。相关性不代表因果，需人工审批和新仿真验证。",
    )
