"""Read existing MOOSE assets without mutating their source workspace."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from app.registry import SimulationRegistry

PARAMETERS = ["preexp_scale", "ER_scale", "tbegin1_shift", "tbegin2_shift", "pyrolysis_heat_scale", "cpv_front_scale", "cpv_mid_scale", "kv_front_scale", "kv_mid_scale", "rho_reactive_scale"]

def _document_id(path: Path) -> str:
    return hashlib.sha256(str(path).encode()).hexdigest()[:20]

def ingest(source: Path, registry: SimulationRegistry) -> dict[str, int]:
    if not source.exists(): raise FileNotFoundError(f"MOOSE source directory not found: {source}")
    registry.reset()
    manifest = pd.read_csv(source / "sampling_manifest_layer1_lhs.csv")
    status = pd.read_csv(source / "batch_run_status_layer1_lhs.csv")
    ranking = pd.read_csv(source / "layer1_lhs_temperature_ranking.csv")
    status_by_case = status.set_index("case_id").to_dict("index")
    metrics_by_case = ranking.set_index("case_id").to_dict("index")
    cases = 0; metrics = 0; documents = 0
    for row in manifest.to_dict("records"):
        case_id = str(row["case_id"])
        run = status_by_case.get(case_id, {})
        params = {name: float(row[name]) for name in PARAMETERS}
        registry.upsert_case(case_id, params, str(run.get("status", "unknown")), int(run["return_code"]) if pd.notna(run.get("return_code")) else None, float(run["elapsed_s"]) if pd.notna(run.get("elapsed_s")) else None, str(source))
        cases += 1
        for name, value in metrics_by_case.get(case_id, {}).items():
            if name in PARAMETERS or name in {"status", "return_code", "elapsed_s"} or pd.isna(value): continue
            if isinstance(value, (int, float)):
                registry.upsert_metric(case_id, name, float(value)); metrics += 1
        case_dir = source / "cases_layer1_lhs" / case_id
        if case_dir.exists():
            for artifact in case_dir.glob("*.csv"): registry.add_artifact(case_id, "curve_or_output", str(artifact))
            for artifact in case_dir.glob("*.i"): registry.add_artifact(case_id, "input_deck", str(artifact))
    candidates = [source / "layer1_lhs_analysis_summary.txt", source / "batch_run_layer1_lhs.log", source / "batch_run_status_layer1_lhs.csv", source / "run_batch_layer1_cases.py", source / "generate_lhs_layer1_main_cases.py", source / "case1_fiat_walltemp_nominal.i"]
    candidates.extend(sorted((source / "cases_layer1_lhs").glob("case_*/*.i"))[:10] if (source / "cases_layer1_lhs").exists() else [])
    for path in candidates:
        if not path.exists(): continue
        content = path.read_text(encoding="utf-8", errors="replace")[:30000]
        if content.strip():
            source_type = "run_log" if path.suffix == ".log" else "input_deck" if path.suffix == ".i" else "script" if path.suffix == ".py" else "run_status" if path.suffix == ".csv" else "report"
            registry.add_document(_document_id(path), source_type, str(path), content); documents += 1
    return {"cases": cases, "metrics": metrics, "documents": documents}
