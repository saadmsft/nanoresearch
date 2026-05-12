"""Shared schemas for the User Profile, Skill, and Memory stores.

Field structure mirrors paper §12 (the concrete Profile A/B/C exemplars), so
that LLM-distilled outputs can be validated against the same JSON shape used
by the authors.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


# ===========================================================================
# User profile (paper §3.1, §12: Profile A/B/C)
# ===========================================================================


class RiskPreference(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class StrictnessLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class UserProfile(BaseModel):
    """Persistent user 𝒰 that grounds every planning decision (paper §3.1).

    Stored at ``data/users/<user_id>/profile.json``. Updated rarely (after the
    initial interactive intake or on explicit user edit).
    """

    user_id: str
    archetype: str = Field(
        ...,
        description="Persona archetype (e.g. ai4science_journal, nlp_conference, high_novelty_exploratory).",
    )
    domain: str = Field(..., description="Primary research domain (NLP, CV, Time Series, …).")
    research_preference: str = ""
    method_preference: str = ""
    risk_preference: RiskPreference = RiskPreference.MODERATE
    baseline_strictness: StrictnessLevel = StrictnessLevel.HIGH
    resource_budget: str = Field("", description="Free-form, e.g. '1× A100 80GB, 5 days'.")
    feasibility_bias: str = ""
    writing_tone: str = ""
    claim_strength: str = ""
    section_organization: str = ""
    venue_style: str = ""
    latex_template: str = "conference_template"
    figure_style: str = ""
    caption_style: str = ""
    priority_feedback: str = ""
    unacceptable_errors: str = ""
    router_hints: str = Field(
        "",
        description="Hints used by the Orchestrator when ranking skills/memories.",
    )
    persona_brief: str = Field("", description="One-paragraph free-form persona summary.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ===========================================================================
# Skills (paper §3.3, §12 Skill exemplars)
# ===========================================================================


SkillType = Literal[
    "planning_and_execution_rule",
    "experiment_design_rule",
    "coding_pattern",
    "debugging_strategy",
    "reviewer_alignment_rule",
    "writing_and_planning_rule",
    "evaluation_and_writing_rule",
    "ideation_and_writing_rule",
    "other",
]


class Skill(BaseModel):
    """Compact procedural rule reusable across projects (paper §3.3.1).

    Skills are *project-agnostic*: they describe **how** to do something well
    (e.g. "design one-factor-at-a-time ablations"). Project-specific facts
    belong in :class:`Memory` instead.
    """

    skill_id: str
    skill_type: SkillType = "other"
    name: str
    when_to_apply: str = ""
    procedure: str = ""
    planning_effect: str = ""
    coding_effect: str = ""
    writing_effect: str = ""
    analysis_effect: str = ""
    review_check: str = ""
    do_not: str = ""

    tags: list[str] = Field(default_factory=list)
    usage_count: int = 0
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def searchable_text(self) -> str:
        """Concatenated text used for keyword / embedding scoring."""
        parts = [
            self.name,
            self.when_to_apply,
            self.procedure,
            self.planning_effect,
            self.coding_effect,
            self.writing_effect,
            self.analysis_effect,
            self.review_check,
            " ".join(self.tags),
        ]
        return "\n".join(p for p in parts if p)


# ===========================================================================
# Memory (paper §3.3, §12 Memory exemplars)
# ===========================================================================


MemoryType = Literal[
    "project_context",
    "decision_history",
    "promising_direction",
    "writing_context",
    "failed_hypothesis",
    "successful_outcome",
    "user_constraint",
    "other",
]


class Memory(BaseModel):
    """Project- and user-specific experience entry (paper §3.3.1).

    Memories are *project-specific*: they record what happened on a given
    topic / blueprint. The retrieval mechanism applies **strict condition
    matching** (paper §3.3.1) so they only surface in comparable settings.
    """

    memory_id: str
    memory_type: MemoryType = "other"
    source_stage: str = ""
    topic_scope: str = ""
    content: str = ""

    retrieval_rationale: str = ""
    planning_implication: str = ""
    coding_implication: str = ""
    analysis_implication: str = ""
    writing_implication: str = ""
    failure_mode_to_avoid: str = ""

    tags: list[str] = Field(default_factory=list)
    user_id: str | None = None
    project_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def searchable_text(self) -> str:
        parts = [
            self.topic_scope,
            self.content,
            self.planning_implication,
            self.coding_implication,
            self.analysis_implication,
            self.writing_implication,
            self.failure_mode_to_avoid,
            " ".join(self.tags),
        ]
        return "\n".join(p for p in parts if p)
