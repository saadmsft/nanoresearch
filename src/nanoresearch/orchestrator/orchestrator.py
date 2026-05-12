"""The Orchestrator 𝒪.

Drives one stage at a time::

    1. Retrieve ``𝒮_C, ℳ_C`` for this user + context  (paper Eq. 12)
    2. Call the stage to produce a high-level plan + artefacts
    3. Optionally enqueue user feedback for SDPO training            (Eq. 14-15)
    4. Distill new Skills/Memories from the trajectory and persist    (Eq. 13)
    5. Optionally trigger an SDPO update on accumulated feedback

The Orchestrator is **stage-agnostic** — it just dispatches to whichever
:class:`Stage` subclass the caller asked to run. Stage-specific behaviour
lives in Phases 4-6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import get_settings
from ..llm import LLMRouter
from ..logging import RunManifest, get_logger
from ..schemas import Memory, Skill, UserProfile
from ..stores import (
    DistilledArtefacts,
    MemoryStore,
    ProfileStore,
    SkillBank,
    distill,
)
from .feedback import FeedbackQueue
from .stage import Stage, StageContext, StageResult, StageStatus
from .trajectory import EventKind

if TYPE_CHECKING:  # pragma: no cover
    from ..planner import Planner

_log = get_logger(__name__)


@dataclass
class OrchestratorOutcome:
    """One stage's full lifecycle result."""

    stage_name: str
    status: StageStatus
    result: StageResult
    new_skills: list[Skill]
    new_memories: list[Memory]


class Orchestrator:
    """Central self-evolution loop coordinator.

    The Orchestrator holds references to all per-user stores and the LLM
    router. It is **stateless across users**: pass the active ``user_id``
    when running stages so the right Skill/Memory directories and LoRA
    adapter are wired in.
    """

    def __init__(
        self,
        *,
        router: LLMRouter,
        profile_store: ProfileStore | None = None,
        manifest: RunManifest | None = None,
        planner: Planner | None = None,
        feedback_queue: FeedbackQueue | None = None,
    ) -> None:
        s = get_settings()
        self.router = router
        self.profile_store = profile_store or ProfileStore(s.lora_adapters_dir)
        self.manifest = manifest
        self._planner = planner
        self.feedback_queue = feedback_queue or FeedbackQueue()

    # ============================================================== user
    def get_user(self, user_id: str) -> UserProfile:
        profile = self.profile_store.load(user_id)
        if profile is None:
            raise FileNotFoundError(
                f"No profile for user {user_id!r}. Create it with `nanoresearch init-user` first."
            )
        return profile

    def skill_bank(self, user_id: str) -> SkillBank:
        return SkillBank(self.profile_store.skills_dir(user_id))

    def memory_store(self, user_id: str) -> MemoryStore:
        return MemoryStore(self.profile_store.memories_dir(user_id))

    # ============================================================== retrieval (Eq. 12)
    def retrieve(
        self,
        *,
        user_id: str,
        context_text: str,
        topic_scope: str | None,
        stage: Stage,
    ) -> tuple[list[Skill], list[Memory]]:
        sb = self.skill_bank(user_id)
        ms = self.memory_store(user_id)
        skills = sb.retrieve(
            context_text, k=stage.top_k_skills, tags=list(stage.retrieval_tags)
        )
        memories = ms.retrieve(
            context_text,
            k=stage.top_k_memories,
            tags=list(stage.retrieval_tags),
            topic_scope=topic_scope,
        )
        for s in skills:
            sb.increment_usage(s.skill_id)
        _log.debug(
            "retrieve",
            stage=stage.name,
            user_id=user_id,
            n_skills=len(skills),
            n_memories=len(memories),
        )
        return skills, memories

    # ============================================================== run
    def run_stage(
        self,
        stage: Stage,
        *,
        user_id: str,
        topic: str,
        project_id: str,
        previous_outputs: dict[str, object] | None = None,
        user_feedback: str | None = None,
        retrieval_query: str | None = None,
        topic_scope: str | None = None,
    ) -> OrchestratorOutcome:
        """Run one stage end-to-end.

        Steps: retrieve → execute → record manifest → enqueue feedback →
        distill → persist new skills/memories → return outcome.
        """
        profile = self.get_user(user_id)

        # 1. Pre-stage retrieval (Eq. 12)
        query = retrieval_query or f"{topic}\n{stage.name}"
        skills, memories = self.retrieve(
            user_id=user_id,
            context_text=query,
            topic_scope=topic_scope or topic,
            stage=stage,
        )

        if self.manifest:
            self.manifest.stage(stage.name, "started", topic=topic, user_id=user_id)

        ctx = StageContext(
            stage_name=stage.name,
            topic=topic,
            user_profile=profile,
            project_id=project_id,
            previous_outputs=dict(previous_outputs or {}),
            retrieved_skills=skills,
            retrieved_memories=memories,
        )

        # 2. Execute stage
        try:
            result = stage.run(ctx, self)
        except Exception as exc:  # noqa: BLE001 - we want all failures here
            _log.exception("stage_crashed", stage=stage.name)
            if self.manifest:
                self.manifest.stage(stage.name, "failed", error=str(exc))
            raise

        # 3. Manifest record
        if self.manifest:
            self.manifest.stage(
                stage.name,
                result.status.value,
                summary=result.summary[:500],
            )

        # 4. Feedback intake (paper §3.3.2) — either the stage already
        #    collected feedback, or the caller passed it from a UI prompt.
        fb_text = result.feedback or user_feedback
        if (
            fb_text
            and result.planner_prompt_messages is not None
            and result.planner_response is not None
        ):
            self.feedback_queue.add(
                user_id=user_id,
                stage=stage.name,
                prompt_messages=result.planner_prompt_messages,
                response=result.planner_response,
                feedback=fb_text,
            )
            result.trajectory.add(
                EventKind.NOTE,
                "feedback_enqueued",
                detail=fb_text,
                stage=stage.name,
            )

        # 5. Distillation + persistence (Eq. 13). Only on success.
        new_skills: list[Skill] = []
        new_memories: list[Memory] = []
        if result.status is StageStatus.SUCCESS:
            distilled: DistilledArtefacts = distill(
                router=self.router,
                trajectory_summary=result.trajectory.summarise(),
                stage=stage.name,
                user_id=user_id,
                project_id=project_id,
            )
            sb = self.skill_bank(user_id)
            ms = self.memory_store(user_id)
            for s in distilled.skills:
                new_skills.append(sb.add(s))
            for m in distilled.memories:
                new_memories.append(ms.add(m))
            if new_skills or new_memories:
                _log.info(
                    "stage_distilled",
                    stage=stage.name,
                    user_id=user_id,
                    new_skills=len(new_skills),
                    new_memories=len(new_memories),
                )

        return OrchestratorOutcome(
            stage_name=stage.name,
            status=result.status,
            result=result,
            new_skills=new_skills,
            new_memories=new_memories,
        )

    # ============================================================== SDPO trigger
    def maybe_train_planner(
        self,
        user_id: str,
        *,
        min_examples: int = 1,
    ) -> dict[str, float] | None:
        """Flush the user's feedback queue and run one SDPO round.

        Skipped if (a) no planner attached, or (b) fewer than ``min_examples``
        feedback records buffered.
        """
        pending = self.feedback_queue.pending_for(user_id)
        if len(pending) < min_examples:
            return None
        if self._planner is None:
            _log.warning(
                "sdpo_skipped_no_planner", user_id=user_id, pending=len(pending)
            )
            return None

        examples = [r.example for r in pending]
        summary = self._planner.update_from_feedback(user_id=user_id, examples=examples)
        self.feedback_queue.drain(user_id)
        return summary
