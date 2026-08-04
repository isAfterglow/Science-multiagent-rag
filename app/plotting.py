"""Whitelisted scientific plots with traceable artifact metadata."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.analysis import parameter_correlation, top_cases
from app.config import ARTIFACTS_DIR
from app.registry import SimulationRegistry


def _save(fig, chart_type: str, source: str, filters: dict[str, Any], sample_size: int, limitations: list[str]) -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_id = f"plot-{uuid.uuid4().hex[:12]}"
    path = ARTIFACTS_DIR / f"{artifact_id}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {"artifact_id": artifact_id, "chart_type": chart_type, "image_path": str(path), "data_source": source, "filters": filters, "sample_size": sample_size, "limitations": limitations}


def plot_parameter_scatter(registry: SimulationRegistry, parameter: str, metric: str) -> dict:
    rows = registry.metric_frame(metric)
    if not rows or parameter not in rows[0]: raise ValueError(f"Unknown parameter or metric: {parameter}, {metric}")
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.2)); ax.scatter(frame[parameter], frame[metric], color="#0f766e")
    slope, intercept = __import__("numpy").polyfit(frame[parameter], frame[metric], 1)
    ax.plot(frame[parameter].sort_values(), slope * frame[parameter].sort_values() + intercept, color="#b42318")
    ax.set(xlabel=parameter, ylabel=metric, title=f"{parameter} vs {metric}")
    return _save(fig, "parameter_scatter", "simulation_registry.metrics + simulation_cases.parameters", {"parameter": parameter, "metric": metric}, len(frame), ["拟合线描述历史样本相关性，不证明因果关系。"])


def plot_parameter_correlation_bar(registry: SimulationRegistry, metric: str) -> dict:
    result = parameter_correlation(registry, metric); values = result["correlations"]
    series = pd.Series(values).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 4.8)); ax.barh(series.index, series.values, color=["#b42318" if value < 0 else "#075985" for value in series.values])
    ax.axvline(0, color="#667085", linewidth=0.8); ax.set(xlabel="Pearson correlation", title=f"Parameter correlation with {metric}")
    return _save(fig, "parameter_correlation_bar", result["source"], {"metric": metric}, result["sample_size"], ["Pearson 相关性不证明因果关系。"])


def plot_metric_ranking_bar(registry: SimulationRegistry, metric: str, top_k: int = 10) -> dict:
    rows = top_cases(registry, metric, min(max(top_k, 1), 20))["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.bar([row["case_id"] for row in rows], [row["metric_value"] for row in rows], color="#075985")
    ax.tick_params(axis="x", rotation=35); ax.set(ylabel=metric, title=f"Top {len(rows)} lowest {metric}")
    return _save(fig, "metric_ranking_bar", "simulation_registry.metrics", {"metric": metric, "top_k": top_k}, len(rows), ["仅按历史指标排序，未控制其他变量。"])


def plot_exploration_plan_diff(plan: dict, baseline_values: dict[str, float]) -> dict:
    cases = plan.get("cases", [])
    if not cases: raise ValueError("Plan has no cases")
    names = sorted(baseline_values); frame = pd.DataFrame({case["label"]: [case["values"][name] - baseline_values[name] for name in names] for case in cases}, index=names)
    fig, ax = plt.subplots(figsize=(8.0, 4.8)); frame.plot(kind="bar", ax=ax)
    ax.axhline(0, color="#667085", linewidth=0.8); ax.set(ylabel="candidate - baseline", title="Exploration plan parameter deltas")
    return _save(fig, "exploration_plan_diff", "simulation_plan_draft + simulation_registry", {"plan_id": plan.get("plan_id", "")}, len(cases), ["显示参数扰动，不代表预测性能改善。"])


def plot_execution_temperature_timeseries(case_id: str, case_dir: Path) -> dict:
    """Plot raw MOOSE temperature outputs from one completed isolated case."""
    files = sorted(case_dir.glob("*pointvalues*.csv"))
    if not files:
        raise ValueError(f"No pointvalues CSV for completed case: {case_id}")
    frame = pd.read_csv(files[0])
    if "time" not in frame:
        raise ValueError("Pointvalues CSV has no time column")
    columns = [name for name in frame.columns if name.startswith("T")]
    if not columns:
        raise ValueError("Pointvalues CSV has no temperature columns")
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for name in columns:
        ax.plot(frame["time"], frame[name], label=name)
    ax.set(xlabel="time (s)", ylabel="temperature", title=f"MOOSE output temperature history: {case_id}")
    ax.legend(ncol=2, fontsize=8)
    return _save(fig, "execution_temperature_timeseries", "completed_MOOSE_pointvalues_csv", {"case_id": case_id, "csv": str(files[0])}, len(frame), ["这是单个仿真 case 的原始输出，不是实验验证，也不能单独证明参数改善。"])
