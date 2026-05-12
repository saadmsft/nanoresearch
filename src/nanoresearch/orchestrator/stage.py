"""Stage abstraction.

Every stage (Ideation, Planning, Coding, Debug, Analysis, Writing, Review)
subclasses :class:`Stage` and implements :meth:`Stage.run`. The Orchestrator
takes care of pre-stage retrieval, post-stage distillation/persistence, and
feedback intake.

This module defines the framework; stage-specific subclasses are added in
Phases 4-6.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ..schemas import Memory, Skill, UserProfile
from .trajectory import Trajectory

if TYPE_CHECKING:  # pragma: no cover
    from .orchestrator import Orchestrator


class StageStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageContext:
    """Inputs the Orchestrator passes to a stage.

    ``previous_outputs`` accumulates artefacts from prior stages so later
    stages can reference, e.g., the ideation hypothesis or the executed
    blueprint.
    """

    stage_name: str
    topic: str
    user_profile: UserProfile
    project_id: str
    previous_outputs: dict[str, Any] = field(default_factory=dict)
    retrieved_skills: list[Skill] = field(default_factory=list)
    retrieved_memories: list[Memory] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """What a stage hands back to the Orchestrator."""

    status: StageStatus
    artefacts: dict[str, Any] = field(default_factory=dict)
    trajectory: Trajectory = field(default_factory=lambda: Trajectory(stage="unknown"))
    summary: str = ""
    # Optional: if the stage already collected feedback (e.g., during peer-review
    # correction loops), include it so the Orchestrator can enqueue an SDPO
    # example without a second prompt.
    feedback: str | None = None
    # Optional: the planner-facing prompt + response that produced the
    # high-level plan, needed for SDPO training. Stages that *use* the
    # planner should populate this.
    planner_prompt_messages: list[dict[str, str]] | None = None
    planner_response: str | None = None


class Stage(ABC):
    """Abstract base for every pipeline stage."""

    #: Human-readable name used in logs + manifest + Skill/Memory routing.
    name: str = "abstract"

    #: Retrieval tags fed into the SkillBank/MemoryStore scoring at pre-stage
    #: retrieval time. Override per-subclass.
    retrieval_tags: tuple[str, ...] = ()

    #: Number of skills/memories to retrieve before running the stage.
    top_k_skills: int = 5
    top_k_memories: int = 5

    @abstractmethod
    def run(
        self,
        context: StageContext,
        orchestrator: Orchestrator,
    ) -> StageResult:
        """Execute the stage and return a :class:`StageResult`."""
