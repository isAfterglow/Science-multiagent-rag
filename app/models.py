from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SimulationCase(BaseModel):
    case_id: str
    parameters: dict[str, float]
    status: str = "unknown"
    return_code: int | None = None
    elapsed_s: float | None = None


class MetricRecord(BaseModel):
    case_id: str
    metric_name: str
    metric_value: float


class EvidenceCard(BaseModel):
    claim: str
    source_type: str
    source_path: str
    excerpt: str
    score: float = Field(ge=0)
    conditions: list[str] = Field(default_factory=list)
    retrieval: dict = Field(default_factory=dict)


class AnalysisEvidence(BaseModel):
    claim: str
    metric: str
    source: str
    result: dict
    limitations: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    issue_type: str
    message: str
    severity: str = "info"


class ReviewDecision(BaseModel):
    approved: bool
    reasons: list[str] = Field(default_factory=list)
    requires_revision: bool = False


class GroundedStatement(BaseModel):
    """A user-visible statement with an explicit, inspectable evidence link."""
    text: str = Field(min_length=1, max_length=1600)
    evidence_kind: str = Field(pattern="^(analysis|document|limitation)$")
    source_path: str
    chunk_id: str = ""
    start_line: int | None = None
    end_line: int | None = None
    support: str = Field(min_length=1, max_length=900)


class EvidenceRequirement(BaseModel):
    kind: Literal["registry_analysis", "report", "input_deck", "run_log", "run_status", "script", "paper", "scan_report"]
    reason: str = Field(min_length=3, max_length=240)
    required: bool = True


class ParameterSet(BaseModel):
    label: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,32}$")
    values: dict[str, float]
    surrogate_prediction: float | None = None


class SimulationPlan(BaseModel):
    plan_id: str
    target_metric: str
    template_sha256: str
    cases: list[ParameterSet] = Field(min_length=1, max_length=5)
    estimated_seconds: float = Field(gt=0)
    max_case_timeout_seconds: int = Field(ge=30, le=600)
    rationale: str = Field(min_length=10, max_length=1200)
