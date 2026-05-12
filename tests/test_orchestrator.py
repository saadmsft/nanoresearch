"""Unit tests for the Orchestrator core loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanoresearch.llm import (
    ChatMessage,
    CompletionResult,
    LLMBackend,
    LLMRouter,
    Role,
)
from nanoresearch.logging import RunManifest
from nanoresearch.orchestrator import (
    FeedbackQueue,
    Orchestrator,
    Stage,
    StageContext,
    StageResult,
    StageStatus,
    Trajectory,
)
from nanoresearch.orchestrator.trajectory import EventKind
from nanoresearch.schemas import UserProfile
from nanoresearch.stores import ProfileStore


# --------------------------------------------------------------------- fakes


class ScriptedBackend(LLMBackend):
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage], **_: Any) -> CompletionResult:
        self.calls.append(messages)
        text = self.replies.pop(0) if self.replies else ""
        return CompletionResult(
            text=text,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            finish_reason="stop",
            backend=self.name,
            model="scripted",
            latency_ms=1.0,
        )


class CountingStage(Stage):
    name = "ideation"
    retrieval_tags = ("ideation",)

    def __init__(self, plan_text: str = "draft ideation plan") -> None:
        self.plan_text = plan_text
        self.invocations = 0

    def run(self, context: StageContext, orchestrator: Orchestrator) -> StageResult:
        self.invocations += 1
        traj = Trajectory(stage=self.name)
        traj.action("retrieved", detail=f"skills={len(context.retrieved_skills)}")
        traj.action("planned", detail=self.plan_text)
        traj.outcome("completed", detail="ok")
        return StageResult(
            status=StageStatus.SUCCESS,
            artefacts={"plan": self.plan_text},
            trajectory=traj,
            summary=self.plan_text,
            planner_prompt_messages=[
                {"role": "system", "content": "You plan experiments."},
                {"role": "user", "content": context.topic},
            ],
            planner_response=self.plan_text,
        )


class FailingStage(Stage):
    name = "coding"

    def run(self, context: StageContext, orchestrator: Orchestrator) -> StageResult:
        traj = Trajectory(stage=self.name)
        traj.error("build_failed", detail="missing import")
        return StageResult(
            status=StageStatus.FAILED,
            trajectory=traj,
            summary="build failed",
        )


# --------------------------------------------------------------------- fixtures


def _make_profile_store(tmp_path: Path, user_id: str = "alice") -> tuple[ProfileStore, UserProfile]:
    store = ProfileStore(tmp_path)
    profile = UserProfile(
        user_id=user_id,
        archetype="ai4science_journal",
        domain="Time Series",
    )
    store.save(profile)
    return store, profile


def _distill_reply() -> str:
    return json.dumps(
        {
            "skills": [
                {
                    "skill_type": "planning_and_execution_rule",
                    "name": "Always reproduce baselines first",
                    "when_to_apply": "before proposing a new method",
                    "procedure": "set up baselines on the same split before any change",
                    "planning_effect": "blueprint includes baseline reproduction",
                    "coding_effect": "shared dataloader and training script",
                    "writing_effect": "",
                    "analysis_effect": "comparable numbers across variants",
                    "review_check": "reviewer can verify baseline reproducibility",
                    "do_not": "skip baselines",
                    "tags": ["baselines", "reproducibility"],
                }
            ],
            "memories": [
                {
                    "memory_type": "project_context",
                    "source_stage": "ideation",
                    "topic_scope": "lightweight UCI HAR sensor classification",
                    "content": "Use 1D CNN + GRU + InceptionTime-small as the baseline suite",
                    "retrieval_rationale": "Recurring requirement for this user.",
                    "planning_implication": "Always include all three baselines.",
                    "coding_implication": "",
                    "analysis_implication": "",
                    "writing_implication": "",
                    "failure_mode_to_avoid": "Diverging splits across baselines.",
                    "tags": ["uci_har", "baselines"],
                }
            ],
        }
    )


# --------------------------------------------------------------------- tests


def test_orchestrator_runs_stage_and_persists_distilled_artefacts(tmp_path: Path) -> None:
    store, profile = _make_profile_store(tmp_path)
    backend = ScriptedBackend(replies=[_distill_reply()])
    router = LLMRouter(azure=backend, planner=ScriptedBackend([]))
    orch = Orchestrator(router=router, profile_store=store)

    stage = CountingStage()
    outcome = orch.run_stage(
        stage,
        user_id=profile.user_id,
        topic="UCI HAR lightweight classification",
        project_id="proj-uci-1",
    )

    assert outcome.status is StageStatus.SUCCESS
    assert stage.invocations == 1
    assert outcome.result.artefacts["plan"] == "draft ideation plan"
    assert len(outcome.new_skills) == 1
    assert outcome.new_skills[0].name == "Always reproduce baselines first"
    assert len(outcome.new_memories) == 1
    assert outcome.new_memories[0].topic_scope == "lightweight UCI HAR sensor classification"

    # Persistence: SkillBank + MemoryStore on disk for this user
    sb = orch.skill_bank(profile.user_id)
    ms = orch.memory_store(profile.user_id)
    assert len(sb) == 1
    assert len(ms) == 1


def test_failed_stage_skips_distillation(tmp_path: Path) -> None:
    store, profile = _make_profile_store(tmp_path)
    backend = ScriptedBackend(replies=[])  # would error if distill were called
    router = LLMRouter(azure=backend, planner=ScriptedBackend([]))
    orch = Orchestrator(router=router, profile_store=store)

    outcome = orch.run_stage(
        FailingStage(),
        user_id=profile.user_id,
        topic="topic",
        project_id="p",
    )
    assert outcome.status is StageStatus.FAILED
    assert outcome.new_skills == []
    assert outcome.new_memories == []
    # Backend should never have been called.
    assert backend.calls == []


def test_retrieval_results_passed_to_stage(tmp_path: Path) -> None:
    store, profile = _make_profile_store(tmp_path)
    # Pre-seed a skill the retrieval should surface.
    sb = ProfileStore(tmp_path).skills_dir(profile.user_id)
    from nanoresearch.schemas import Skill
    from nanoresearch.stores import SkillBank

    bank = SkillBank(sb)
    bank.add(
        Skill(
            skill_id="s-baselines",
            name="Reproduce baselines first",
            procedure="set up baselines with shared split before any change",
            tags=["ideation", "baselines"],
            confidence=0.9,
        )
    )

    backend = ScriptedBackend(replies=[_distill_reply()])
    router = LLMRouter(azure=backend, planner=ScriptedBackend([]))
    orch = Orchestrator(router=router, profile_store=store)

    captured: list[StageContext] = []

    class CaptureStage(CountingStage):
        def run(self, ctx: StageContext, orchestrator: Orchestrator) -> StageResult:
            captured.append(ctx)
            return super().run(ctx, orchestrator)

    orch.run_stage(
        CaptureStage(),
        user_id=profile.user_id,
        topic="reproduce baselines for UCI HAR",
        project_id="p",
    )
    assert len(captured) == 1
    skill_ids = [s.skill_id for s in captured[0].retrieved_skills]
    assert "s-baselines" in skill_ids
    # Retrieval should bump usage_count
    bumped = bank.get("s-baselines")
    assert bumped is not None and bumped.usage_count >= 1


def test_feedback_enqueued_for_sdpo(tmp_path: Path) -> None:
    store, profile = _make_profile_store(tmp_path)
    backend = ScriptedBackend(replies=[_distill_reply()])
    router = LLMRouter(azure=backend, planner=ScriptedBackend([]))
    orch = Orchestrator(router=router, profile_store=store)

    outcome = orch.run_stage(
        CountingStage(),
        user_id=profile.user_id,
        topic="UCI HAR",
        project_id="p",
        user_feedback="Prefer simpler methods with smaller param counts.",
    )
    assert outcome.status is StageStatus.SUCCESS
    pending = orch.feedback_queue.pending_for(profile.user_id)
    assert len(pending) == 1
    assert pending[0].example.feedback.startswith("Prefer simpler")
    assert pending[0].example.response == "draft ideation plan"


def test_maybe_train_planner_skipped_without_planner(tmp_path: Path) -> None:
    store, profile = _make_profile_store(tmp_path)
    backend = ScriptedBackend(replies=[_distill_reply()])
    router = LLMRouter(azure=backend, planner=ScriptedBackend([]))
    orch = Orchestrator(router=router, profile_store=store)

    orch.run_stage(
        CountingStage(),
        user_id=profile.user_id,
        topic="t",
        project_id="p",
        user_feedback="Be terser.",
    )
    # No planner attached → SDPO step is a no-op and returns None
    assert orch.maybe_train_planner(profile.user_id) is None
    # Queue is NOT drained when no planner is present
    assert len(orch.feedback_queue.pending_for(profile.user_id)) == 1


def test_manifest_records_stage_lifecycle(tmp_path: Path) -> None:
    store, profile = _make_profile_store(tmp_path)
    backend = ScriptedBackend(replies=[_distill_reply()])
    router = LLMRouter(azure=backend, planner=ScriptedBackend([]))
    manifest = RunManifest(tmp_path / "runs")
    orch = Orchestrator(router=router, profile_store=store, manifest=manifest)

    orch.run_stage(
        CountingStage(), user_id=profile.user_id, topic="t", project_id="p"
    )
    events = [json.loads(l) for l in manifest.events_path.read_text().splitlines()]
    stage_events = [e for e in events if e["event"] == "stage"]
    statuses = [e["status"] for e in stage_events]
    assert "started" in statuses
    assert "success" in statuses


def test_trajectory_summarise_includes_events() -> None:
    t = Trajectory(stage="ideation")
    t.action("planned", detail="three hypotheses generated")
    t.critique("blueprint critique", detail="missing ablation X")
    t.outcome("completed", detail="ok")
    s = t.summarise()
    assert "ideation" in s
    assert "planned" in s
    assert "blueprint critique" in s
    assert "completed" in s


def test_feedback_queue_drain(tmp_path: Path) -> None:
    q = FeedbackQueue()
    for i in range(3):
        q.add(
            user_id="alice",
            stage="ideation",
            prompt_messages=[{"role": "user", "content": f"q{i}"}],
            response=f"r{i}",
            feedback=f"fb{i}",
        )
    assert len(q.pending_for("alice")) == 3
    drained = q.drain("alice")
    assert len(drained) == 3
    assert len(q.pending_for("alice")) == 0
