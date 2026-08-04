"""Pydantic contracts for LLM assistance; no free-form cross-agent messages."""
from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, Field

class PlanHint(BaseModel):
    task_type: str = Field(pattern="^(knowledge|simulation_analysis|mixed)$")
    retrieval_query: str = Field(min_length=1, max_length=300)
    analysis_metric: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=300)


class RoutePlan(BaseModel):
    """Advisory semantic route proposed by an LLM, never an execution grant."""
    task_type: Literal["knowledge", "simulation_analysis", "mixed"]
    needs_registry_analysis: bool
    required_sources: list[Literal["report", "input_deck", "run_log", "run_status", "script", "paper", "scan_report"]] = Field(default_factory=list, max_length=4)
    needs_experiment: bool
    retrieval_query: str = Field(min_length=1, max_length=300)
    analysis_metric: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=300)

class EvidenceSummary(BaseModel):
    summary: str = Field(min_length=1, max_length=900)
    limitations: list[str] = Field(default_factory=list, max_length=4)


class ResearchClaim(BaseModel):
    """A non-authoritative synthesis claim tied to retrieved chunks."""
    text: str = Field(min_length=1, max_length=420)
    evidence_indexes: list[int] = Field(min_length=1, max_length=4)
    limitations: list[str] = Field(default_factory=list, max_length=3)


class ResearchSynthesis(BaseModel):
    claims: list[ResearchClaim] = Field(default_factory=list, max_length=4)
    conflicts: list[str] = Field(default_factory=list, max_length=3)
    evidence_gap: str = Field(default="", max_length=360)


class PlannerProposal(BaseModel):
    """Advice only; deterministic code constructs the executable plan."""
    target_metric: str = Field(min_length=1, max_length=64)
    focus_parameters: list[str] = Field(default_factory=list, max_length=3)
    rationale: str = Field(min_length=1, max_length=600)
    requires_human_approval: bool


class SemanticCritiqueItem(BaseModel):
    issue_type: Literal["overreach", "conflict", "missing_evidence", "causality"]
    severity: Literal["info", "warning", "error"]
    message: str = Field(min_length=1, max_length=420)
    evidence_indexes: list[int] = Field(default_factory=list, max_length=4)


class SemanticCritique(BaseModel):
    issues: list[SemanticCritiqueItem] = Field(default_factory=list, max_length=4)
