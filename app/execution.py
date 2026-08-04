"""Approved-plan executor. It never writes to the historical MOOSE workspace."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import RUNS_DIR
from app.registry import SimulationRegistry
from app.simulation_plan import PlanStore, build_input_deck

def _sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def _prepare_case(run_dir: Path, case: dict, mpi_processes: int) -> Path:
    case_dir = run_dir / case["label"]
    case_dir.mkdir(parents=True, exist_ok=False)
    deck = case_dir / "case1_fiat_walltemp_nominal.i"
    deck.write_text(build_input_deck(case["values"]), encoding="utf-8")
    from app.config import RUN_SCRIPT_PATH
    # Preserve the historical launcher and only specialize its isolated copy.
    run_script = (case_dir / "run.sh")
    shutil.copy2(RUN_SCRIPT_PATH, run_script)
    run_script.write_text(run_script.read_text(encoding="utf-8").replace("mpirun -n 4", f"mpirun -n {mpi_processes}"), encoding="utf-8")
    (case_dir / "params.json").write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "manifest.json").write_text(json.dumps({"input_sha256": _sha256(deck), "source_template": str(RUN_SCRIPT_PATH), "mpi_processes": mpi_processes}, ensure_ascii=False, indent=2), encoding="utf-8")
    return case_dir

def _extract_summary(case_dir: Path) -> dict[str, Any]:
    files = sorted(case_dir.glob("*pointvalues*.csv"))
    if not files: return {}
    frame = pd.read_csv(files[0])
    numeric = frame.select_dtypes(include="number")
    if numeric.empty: return {}
    output_metrics = {f"final_{column}": round(float(value), 6) for column, value in numeric.iloc[-1].items()}
    if "time" in numeric:
        output_metrics["final_time_s"] = output_metrics.pop("final_time")
    return {"observed_numeric_max": round(float(numeric.max().max()), 6), "observed_numeric_min": round(float(numeric.min().min()), 6), "output_metrics": output_metrics}


def record_case_outputs(case_id: str, case_dir: Path, registry: SimulationRegistry) -> dict[str, Any]:
    """Register raw simulation output features, never relabeling them as RMSE."""
    summary = _extract_summary(case_dir)
    for name, value in summary.get("output_metrics", {}).items():
        registry.upsert_metric(case_id, name, value)
    log_path = case_dir / "log"
    if log_path.exists(): registry.add_artifact(case_id, "run_log", str(log_path))
    for path in sorted(case_dir.glob("*.csv")):
        registry.add_artifact(case_id, "simulation_output", str(path))
    return summary

def _failure_category(stdout: str, stderr: str, return_code: int) -> str:
    text = (stdout + "\n" + stderr).lower()
    if "operation not permitted" in text or "unable to listen on port" in text:
        return "environment_blocked"
    if return_code != 0: return "simulation_or_runtime_failure"
    return "none"

def execute_approved_plan(plan_id: str, store: PlanStore, registry: SimulationRegistry, *, dry_run: bool = True, mpi_processes: int = 4) -> dict[str, Any]:
    if mpi_processes not in {1, 2, 4}:
        raise ValueError("mpi_processes must be one of 1, 2, or 4")
    record = store.get(plan_id)
    if not record: raise KeyError(plan_id)
    if record["status"] != "approved": raise PermissionError("Only an approved plan may execute.")
    plan = record["plan"]
    run_dir = RUNS_DIR / plan_id
    if run_dir.exists(): raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    for case in plan["cases"]:
        case_dir = _prepare_case(run_dir, case, mpi_processes)
        started = time.perf_counter()
        if dry_run:
            result = {"label": case["label"], "status": "preview", "elapsed_s": 0.0, "case_dir": str(case_dir), "input_sha256": _sha256(case_dir / "case1_fiat_walltemp_nominal.i"), "command": ["bash", "run.sh"], "mpi_processes": mpi_processes}
        else:
            try:
                completed = subprocess.run(["bash", "run.sh"], cwd=case_dir, capture_output=True, text=True, timeout=int(plan["max_case_timeout_seconds"]))
                (case_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8"); (case_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
                category = _failure_category(completed.stdout, completed.stderr, completed.returncode)
                status = "ok" if completed.returncode == 0 else category
                result = {"label": case["label"], "status": status, "return_code": completed.returncode, "elapsed_s": round(time.perf_counter() - started, 3), "case_dir": str(case_dir), "input_sha256": _sha256(case_dir / "case1_fiat_walltemp_nominal.i"), "mpi_processes": mpi_processes, "failure_category": category, "summary": _extract_summary(case_dir)}
            except subprocess.TimeoutExpired:
                result = {"label": case["label"], "status": "timeout", "elapsed_s": round(time.perf_counter() - started, 3), "case_dir": str(case_dir), "input_sha256": _sha256(case_dir / "case1_fiat_walltemp_nominal.i"), "mpi_processes": mpi_processes}
        results.append(result)
        registry.upsert_case(f"{plan_id}-{case['label']}", case["values"], result["status"], result.get("return_code"), result["elapsed_s"], str(case_dir))
        registry.add_artifact(f"{plan_id}-{case['label']}", "input_deck", str(case_dir / "case1_fiat_walltemp_nominal.i"))
        registry.add_artifact(f"{plan_id}-{case['label']}", "execution_manifest", str(case_dir / "manifest.json"))
        if result["status"] == "ok":
            result["summary"] = record_case_outputs(f"{plan_id}-{case['label']}", case_dir, registry)
    blocked = sum(item["status"] == "environment_blocked" for item in results)
    conclusion = "这是执行预览，未启动 MOOSE。" if dry_run else ("真实执行被运行环境阻断；输入、审批和日志均已保留，可在允许 MPI socket 的环境重试。" if blocked else "真实执行已完成；新结果已写入仿真 Registry，仍需与实验/高保真指标复核。")
    review = {"plan_id": plan_id, "dry_run": dry_run, "mpi_processes": mpi_processes, "total_cases": len(results), "successful_cases": sum(item["status"] == "ok" for item in results), "preview_cases": sum(item["status"] == "preview" for item in results), "environment_blocked_cases": blocked, "results": results, "conclusion": conclusion}
    (run_dir / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    store.save_result(plan_id, review)
    return review
