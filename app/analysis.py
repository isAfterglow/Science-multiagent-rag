"""Read-only quantitative tools. All numeric conclusions come from these tools."""
from __future__ import annotations

import pandas as pd

from app.registry import SimulationRegistry

def top_cases(registry: SimulationRegistry, metric: str, limit: int = 5) -> dict:
    rows = registry.top_cases(metric, limit)
    return {"metric": metric, "order": "ascending", "rows": rows, "source": "simulation_registry.metrics"}

def parameter_correlation(registry: SimulationRegistry, metric: str) -> dict:
    rows = registry.metric_frame(metric)
    if len(rows) < 3: raise ValueError(f"Not enough cases for metric: {metric}")
    frame = pd.DataFrame(rows)
    correlations = frame.drop(columns=["case_id"]).corr(numeric_only=True)[metric].drop(metric).sort_values()
    return {"metric": metric, "sample_size": len(frame), "method": "Pearson correlation", "correlations": {key: round(float(value), 6) for key, value in correlations.items()}, "source": "simulation_registry.metrics + simulation_cases.parameters"}

def case_detail(registry: SimulationRegistry, case_id: str) -> dict:
    detail = registry.case_detail(case_id)
    if detail is None: raise ValueError(f"Unknown case: {case_id}")
    return detail

def failed_cases(registry: SimulationRegistry) -> dict:
    rows = [row for row in registry.cases() if row["status"] != "ok"]
    return {"count": len(rows), "rows": rows, "source": "simulation_registry.simulation_cases"}
