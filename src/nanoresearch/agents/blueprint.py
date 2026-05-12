"""Schemas for the **experiment blueprint** ``ℬ`` and intermediate ideation
artefacts.

The blueprint is the primary artefact handed from Stage I (Planning) to
Stage II (Coding & Execution).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..literature import Evidence


class Hypothesis(BaseModel):
    """One candidate research hypothesis ``h_k`` from ideation."""

    hypothesis_id: str
    statement: str
    motivation: str = ""
    expected_contribution: str = ""
    related_papers: list[str] = Field(default_factory=list, description="paper_ids cited.")
    novelty_score: float | None = None
    novelty_rationale: str = ""
    closest_prior_work: str = ""


class ProposedMethod(BaseModel):
    name: str
    description: str
    key_components: list[str] = Field(default_factory=list)
    architecture: str = ""


class AblationGroup(BaseModel):
    name: str
    variants: list[str]
    purpose: str = ""


class Blueprint(BaseModel):
    """Experiment blueprint ``ℬ`` produced by Stage I Planning (paper §3.2.1).

    Stage II Coding/Execution consumes this; Stage III Writing organises the
    final paper around it.
    """

    blueprint_id: str
    title: str
    research_question: str
    hypothesis: str
    proposed_method: ProposedMethod
    datasets: list[str]
    baselines: list[str]
    metrics: list[str]
    ablation_groups: list[AblationGroup] = Field(default_factory=list)
    compute_budget: str = ""
    expected_outcome: str = ""
    risks: list[str] = Field(default_factory=list)
    # Audit trail (mostly filled by the peer-review correction loop).
    reviewer_critiques: list[str] = Field(default_factory=list)
    revision_count: int = 0


class IdeationArtefacts(BaseModel):
    """Outputs of the Ideation phase (paper §3.2.1, before Planning)."""

    plan_PI: str = Field(..., description="High-level ideation plan from the planner.")
    retrieved_papers: list[str] = Field(default_factory=list, description="paper_ids surfaced from the literature.")
    evidence: list[Evidence] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    chosen_hypothesis_id: str | None = None


# ---- LLM judge response types ---------------------------------------------


class NoveltyJudgement(BaseModel):
    hypothesis_id: str
    novelty_score: float
    rationale: str
    closest_baseline: str = ""


class BlueprintCritique(BaseModel):
    verdict: Literal["accept", "revise"]
    issues: list[str] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)
