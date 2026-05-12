"""Stage I — Planning phase (paper §3.2.1, Eq. 2-4).

Translates ``h*`` into a JSON-formatted experiment blueprint ``ℬ``, then runs
a peer-review-like correction loop (Eq. 3) until the blueprint passes review
or the retry limit is exhausted.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..llm import AgentRole, ChatMessage, Role
from ..logging import get_logger
from ..orchestrator.stage import Stage, StageContext, StageResult, StageStatus
from ..orchestrator.trajectory import Trajectory
from ._util import extract_json_object, render_memories, render_skills
from .blueprint import Blueprint, BlueprintCritique, Hypothesis
from .prompts import BLUEPRINT_REVIEWER_SYSTEM, PLANNING_SYSTEM

if TYPE_CHECKING:  # pragma: no cover
    from ..orchestrator.orchestrator import Orchestrator

_log = get_logger(__name__)


@dataclass
class PlanningConfig:
    max_review_iterations: int = 3
    # GPT-5.1 reasoning eats tokens before emitting JSON — keep budgets large.
    max_new_tokens_plan: int = 8000
    max_new_tokens_review: int = 4000


class PlanningStage(Stage):
    """Stage I — Planning. Produces a peer-reviewed :class:`Blueprint`."""

    name = "planning"
    retrieval_tags = ("planning", "blueprint", "experiment_design")

    def __init__(self, *, config: PlanningConfig | None = None) -> None:
        self.config = config or PlanningConfig()

    def run(self, context: StageContext, orchestrator: Orchestrator) -> StageResult:
        traj = Trajectory(stage=self.name)

        chosen: Hypothesis | None = context.previous_outputs.get("chosen_hypothesis")
        if chosen is None:
            traj.error("planning_no_hypothesis", "Ideation must run first.")
            return StageResult(
                status=StageStatus.FAILED,
                trajectory=traj,
                summary="Planning aborted: no chosen hypothesis in previous_outputs.",
            )

        # ---- 1. Initial blueprint -----------------------------------
        blueprint = self._initial_blueprint(
            chosen=chosen, context=context, orchestrator=orchestrator, traj=traj
        )
        if blueprint is None:
            return StageResult(
                status=StageStatus.FAILED,
                trajectory=traj,
                summary="Planning failed to produce an initial blueprint.",
            )

        # ---- 2. Peer-review correction loop (Eq. 3) ------------------
        for it in range(self.config.max_review_iterations):
            critique = self._review_blueprint(
                blueprint=blueprint, orchestrator=orchestrator, traj=traj, iteration=it
            )
            if critique is None or critique.verdict == "accept":
                traj.outcome("blueprint_accepted", detail=f"iterations={it}")
                break
            blueprint.reviewer_critiques.extend(critique.issues)
            traj.critique(
                "blueprint_revise",
                detail=f"iter={it} issues={critique.issues} fixes={critique.suggested_fixes}",
            )
            refined = self._refine_blueprint(
                blueprint=blueprint,
                critique=critique,
                chosen=chosen,
                context=context,
                orchestrator=orchestrator,
                traj=traj,
                iteration=it,
            )
            if refined is None:
                traj.error("blueprint_refine_failed", detail=f"iter={it}")
                break
            blueprint = refined
            blueprint.revision_count += 1
        else:
            traj.note_str = "max_review_iterations_reached"

        return StageResult(
            status=StageStatus.SUCCESS,
            artefacts={
                "blueprint": blueprint,
                "chosen_hypothesis": chosen,
            },
            trajectory=traj,
            summary=f"Blueprint '{blueprint.title}' "
            f"with {len(blueprint.baselines)} baselines and "
            f"{len(blueprint.ablation_groups)} ablation groups "
            f"(revisions={blueprint.revision_count}).",
        )

    # ----------------------------------------------------------- internals

    def _initial_blueprint(
        self,
        *,
        chosen: Hypothesis,
        context: StageContext,
        orchestrator: Orchestrator,
        traj: Trajectory,
    ) -> Blueprint | None:
        profile = context.user_profile
        user_msg = (
            f"# Chosen hypothesis (h*)\n{chosen.statement}\n\n"
            f"motivation: {chosen.motivation}\n"
            f"expected_contribution: {chosen.expected_contribution}\n"
            f"closest_prior_work: {chosen.closest_prior_work}\n\n"
            f"# User profile\n"
            f"- archetype: {profile.archetype}\n"
            f"- domain: {profile.domain}\n"
            f"- method_preference: {profile.method_preference}\n"
            f"- baseline_strictness: {profile.baseline_strictness}\n"
            f"- resource_budget: {profile.resource_budget or '(unspecified)'}\n"
            f"- venue_style: {profile.venue_style}\n\n"
            f"# Retrieved skills\n{render_skills(context.retrieved_skills)}\n\n"
            f"# Retrieved memories\n{render_memories(context.retrieved_memories)}\n\n"
            "Produce a complete blueprint following the schema. Return ONLY the JSON object."
        )
        traj.action("planning_llm_call", detail=f"prompt_chars={len(user_msg)}")
        res = orchestrator.router.complete(
            AgentRole.PLANNING,
            [
                ChatMessage(Role.SYSTEM, PLANNING_SYSTEM),
                ChatMessage(Role.USER, user_msg),
            ],
            temperature=0.3,
            max_tokens=self.config.max_new_tokens_plan,
            response_format={"type": "json_object"},
        )
        raw = extract_json_object(res.text)
        if raw is None:
            traj.error("planning_no_json", detail=res.text[:300])
            return None
        raw.setdefault("blueprint_id", f"bp-{uuid.uuid4().hex[:8]}")
        try:
            return Blueprint.model_validate(raw)
        except ValidationError as e:
            traj.error("planning_invalid_blueprint", detail=str(e))
            _log.warning("planning_invalid_blueprint", error=str(e))
            return None

    def _review_blueprint(
        self,
        *,
        blueprint: Blueprint,
        orchestrator: Orchestrator,
        traj: Trajectory,
        iteration: int,
    ) -> BlueprintCritique | None:
        user_msg = (
            f"# Blueprint to review (iteration {iteration})\n"
            f"{blueprint.model_dump_json(indent=2)}\n\n"
            "Return ONLY the JSON object as specified."
        )
        traj.action("review_llm_call", detail=f"iteration={iteration}")
        res = orchestrator.router.complete(
            AgentRole.REVIEW,
            [
                ChatMessage(Role.SYSTEM, BLUEPRINT_REVIEWER_SYSTEM),
                ChatMessage(Role.USER, user_msg),
            ],
            temperature=0.0,
            max_tokens=self.config.max_new_tokens_review,
            response_format={"type": "json_object"},
        )
        raw = extract_json_object(res.text)
        if raw is None:
            traj.error("review_no_json", detail=res.text[:300])
            return None
        try:
            return BlueprintCritique.model_validate(raw)
        except ValidationError as e:
            _log.warning("review_invalid_critique", error=str(e))
            return None

    def _refine_blueprint(
        self,
        *,
        blueprint: Blueprint,
        critique: BlueprintCritique,
        chosen: Hypothesis,
        context: StageContext,
        orchestrator: Orchestrator,
        traj: Trajectory,
        iteration: int,
    ) -> Blueprint | None:
        user_msg = (
            f"# Previous blueprint (revision {blueprint.revision_count})\n"
            f"{blueprint.model_dump_json(indent=2)}\n\n"
            f"# Reviewer issues\n- " + "\n- ".join(critique.issues) + "\n\n"
            f"# Reviewer suggested fixes\n- " + "\n- ".join(critique.suggested_fixes) + "\n\n"
            f"# Original hypothesis\n{chosen.statement}\n\n"
            "Produce a REVISED blueprint that resolves every issue. Keep the same "
            "blueprint_id. Return ONLY the JSON object."
        )
        traj.action("refine_llm_call", detail=f"iteration={iteration}")
        res = orchestrator.router.complete(
            AgentRole.PLANNING,
            [
                ChatMessage(Role.SYSTEM, PLANNING_SYSTEM),
                ChatMessage(Role.USER, user_msg),
            ],
            temperature=0.3,
            max_tokens=self.config.max_new_tokens_plan,
            response_format={"type": "json_object"},
        )
        raw = extract_json_object(res.text)
        if raw is None:
            traj.error("refine_no_json", detail=res.text[:300])
            return None
        raw["blueprint_id"] = blueprint.blueprint_id
        try:
            refined = Blueprint.model_validate(raw)
        except ValidationError as e:
            traj.error("refine_invalid_blueprint", detail=str(e))
            return None
        # Carry forward audit trail.
        refined.reviewer_critiques = blueprint.reviewer_critiques
        refined.revision_count = blueprint.revision_count
        return refined
