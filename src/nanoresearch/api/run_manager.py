"""In-process run-lifecycle manager.

A run is launched in a background thread. As stages emit trajectory events,
the thread pushes them onto a per-run :class:`asyncio.Queue` consumed by the
SSE endpoint. Feedback submitted via HTTP unblocks the next stage.

This keeps the v1 UI simple — no Redis, no Celery — at the cost of a single
process. The :class:`RunManager` interface is the seam we'd swap out for a
multi-process deployment.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from ..agents import (
    AnalysisStage,
    CodingStage,
    IdeationStage,
    PlanningStage,
    WritingStage,
)
from ..llm import LLMRouter
from ..logging import RunManifest, get_logger
from ..orchestrator import Orchestrator
from ..orchestrator.stage import StageStatus
from ..stores import ProfileStore
from .narrator import narrate_event

_log = get_logger(__name__)


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_FEEDBACK = "awaiting_feedback"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunSnapshot:
    """Serializable snapshot of a run for HTTP responses."""

    run_id: str
    user_id: str
    topic: str
    project_id: str
    status: RunStatus
    current_stage: str | None
    stages_completed: list[str]
    last_summary: str
    started_at: datetime
    updated_at: datetime
    error: str | None = None


@dataclass
class RunState:
    """Internal state kept per run while the background thread executes it."""

    run_id: str
    user_id: str
    topic: str
    project_id: str
    status: RunStatus
    snapshot: RunSnapshot
    queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)
    feedback_event: threading.Event = field(default_factory=threading.Event)
    feedback_text: str | None = None
    thread: threading.Thread | None = None
    artefacts: dict[str, Any] = field(default_factory=dict)


class RunManager:
    """Owns every active run in this process."""

    def __init__(
        self,
        *,
        router: LLMRouter,
        profile_store: ProfileStore,
        runs_dir: str | None = None,
    ) -> None:
        self.router = router
        self.profile_store = profile_store
        self.runs_dir = runs_dir
        self._runs: dict[str, RunState] = {}
        self._lock = threading.Lock()
        # Capture the API server's loop the first time we need it, so
        # background threads can hand events back across the boundary.
        self._loop: asyncio.AbstractEventLoop | None = None

    # ============================================================ accessors

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def get(self, run_id: str) -> RunState | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> list[RunSnapshot]:
        with self._lock:
            return [r.snapshot for r in self._runs.values()]

    # ============================================================ start

    def start_run(self, *, user_id: str, topic: str) -> RunSnapshot:
        # Validate the user exists before we spin up.
        if self.profile_store.load(user_id) is None:
            raise LookupError(f"Unknown user_id: {user_id}")

        run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        now = datetime.now(UTC)
        snapshot = RunSnapshot(
            run_id=run_id,
            user_id=user_id,
            topic=topic,
            project_id=f"proj-{run_id[4:]}",
            status=RunStatus.PENDING,
            current_stage=None,
            stages_completed=[],
            last_summary="",
            started_at=now,
            updated_at=now,
        )
        state = RunState(
            run_id=run_id,
            user_id=user_id,
            topic=topic,
            project_id=snapshot.project_id,
            status=RunStatus.PENDING,
            snapshot=snapshot,
        )

        with self._lock:
            self._runs[run_id] = state

        thread = threading.Thread(
            target=self._execute,
            args=(state,),
            name=f"run-{run_id}",
            daemon=True,
        )
        state.thread = thread
        thread.start()
        return snapshot

    # ============================================================ feedback

    def submit_feedback(self, run_id: str, text: str) -> RunSnapshot:
        state = self.get(run_id)
        if state is None:
            raise LookupError(f"Unknown run: {run_id}")
        if state.status is not RunStatus.AWAITING_FEEDBACK:
            raise ValueError(f"Run {run_id} is not awaiting feedback (status={state.status})")
        state.feedback_text = text
        state.feedback_event.set()
        self._emit(state, "feedback_received", {"text": text})
        return state.snapshot

    # ============================================================ orchestration

    def _execute(self, state: RunState) -> None:
        manifest = RunManifest(runs_dir=_resolve_runs_dir(self.runs_dir), run_id=state.run_id)
        orchestrator = Orchestrator(
            router=self.router,
            profile_store=self.profile_store,
            manifest=manifest,
        )
        try:
            self._update_status(state, RunStatus.RUNNING, current_stage="ideation")
            self._emit(state, "run_started", {"topic": state.topic})

            # ----- Stage I: Ideation -------------------------------------
            ideation = IdeationStage()
            ide_outcome = orchestrator.run_stage(
                ideation,
                user_id=state.user_id,
                topic=state.topic,
                project_id=state.project_id,
            )
            self._after_stage(state, "ideation", ide_outcome)
            if ide_outcome.status is not StageStatus.SUCCESS:
                self._fail(state, "Ideation failed")
                return

            self._wait_for_feedback(state, stage="ideation", orchestrator=orchestrator)

            # ----- Stage I: Planning -------------------------------------
            self._update_status(state, RunStatus.RUNNING, current_stage="planning")
            chosen = ide_outcome.result.artefacts.get("chosen_hypothesis")
            planning = PlanningStage()
            plan_outcome = orchestrator.run_stage(
                planning,
                user_id=state.user_id,
                topic=state.topic,
                project_id=state.project_id,
                previous_outputs={"chosen_hypothesis": chosen},
            )
            self._after_stage(state, "planning", plan_outcome)
            if plan_outcome.status is not StageStatus.SUCCESS:
                self._fail(state, "Planning failed")
                return

            self._wait_for_feedback(state, stage="planning", orchestrator=orchestrator)

            blueprint = plan_outcome.result.artefacts.get("blueprint")
            state.artefacts["blueprint"] = blueprint

            # ----- Stage II: Coding + Execution + Debug ------------------
            self._update_status(state, RunStatus.RUNNING, current_stage="coding")
            coding = CodingStage()
            code_outcome = orchestrator.run_stage(
                coding,
                user_id=state.user_id,
                topic=state.topic,
                project_id=state.project_id,
                previous_outputs={"blueprint": blueprint},
            )
            self._after_stage(state, "coding", code_outcome)
            if code_outcome.status is not StageStatus.SUCCESS:
                self._fail(state, "Coding/execution failed after debug retries")
                return

            execution = code_outcome.result.artefacts.get("execution")
            state.artefacts["execution"] = execution

            # ----- Stage II: Analysis ------------------------------------
            self._update_status(state, RunStatus.RUNNING, current_stage="analysis")
            analysis_stage = AnalysisStage()
            ana_outcome = orchestrator.run_stage(
                analysis_stage,
                user_id=state.user_id,
                topic=state.topic,
                project_id=state.project_id,
                previous_outputs={"blueprint": blueprint, "execution": execution},
            )
            self._after_stage(state, "analysis", ana_outcome)
            if ana_outcome.status is not StageStatus.SUCCESS:
                self._fail(state, "Analysis failed")
                return

            analysis = ana_outcome.result.artefacts.get("analysis")
            state.artefacts["analysis"] = analysis

            self._wait_for_feedback(state, stage="analysis", orchestrator=orchestrator)

            # ----- Stage III: Writing + Review + PDF build --------------
            self._update_status(state, RunStatus.RUNNING, current_stage="writing")
            writing = WritingStage()
            wr_outcome = orchestrator.run_stage(
                writing,
                user_id=state.user_id,
                topic=state.topic,
                project_id=state.project_id,
                previous_outputs={
                    "blueprint": blueprint,
                    "analysis": analysis,
                },
            )
            self._after_stage(state, "writing", wr_outcome)
            if wr_outcome.status is not StageStatus.SUCCESS:
                self._fail(state, "Writing failed")
                return

            compiled = wr_outcome.result.artefacts.get("compiled")
            state.artefacts["paper"] = wr_outcome.result.artefacts.get("paper")
            state.artefacts["compiled"] = compiled
            if compiled is not None:
                self._emit(
                    state,
                    "paper_ready",
                    {
                        "compiled": compiled.compiled,
                        "tex_path": compiled.tex_path,
                        "pdf_path": compiled.pdf_path,
                        "compile_error": compiled.compile_error,
                    },
                )

            self._update_status(state, RunStatus.COMPLETED, current_stage=None)
            self._emit(state, "run_completed", {})
        except Exception as exc:  # noqa: BLE001
            _log.exception("run_crashed", run_id=state.run_id)
            self._fail(state, f"{type(exc).__name__}: {exc}")

    def _wait_for_feedback(
        self, state: RunState, *, stage: str, orchestrator: Orchestrator
    ) -> None:
        self._update_status(state, RunStatus.AWAITING_FEEDBACK, current_stage=stage)
        self._emit(state, "awaiting_feedback", {"stage": stage})
        # Wait up to 30 min for feedback; otherwise carry on.
        state.feedback_event.wait(timeout=1800)
        text = state.feedback_text or ""
        state.feedback_text = None
        state.feedback_event.clear()
        if text:
            # Plumb the feedback into the orchestrator's queue. Since stages
            # in v1 don't yet populate planner_prompt/response, we synthesise
            # a minimal pair so SDPO has something to train against later.
            orchestrator.feedback_queue.add(
                user_id=state.user_id,
                stage=stage,
                prompt_messages=[{"role": "user", "content": state.topic}],
                response=state.snapshot.last_summary or "",
                feedback=text,
            )
            self._emit(state, "feedback_enqueued", {"stage": stage})

    # ============================================================ helpers

    def _after_stage(
        self, state: RunState, stage: str, outcome: Any
    ) -> None:
        events = [e for e in outcome.result.trajectory.events]
        for e in events:
            self._emit(
                state,
                "trajectory_event",
                {
                    "stage": stage,
                    "kind": e.kind.value,
                    "label": e.label,
                    "detail": e.detail[:600],
                    "metadata": {k: str(v) for k, v in (e.metadata or {}).items()},
                    "ts": e.ts.isoformat(),
                },
            )
        state.snapshot.stages_completed.append(stage)
        state.snapshot.last_summary = outcome.result.summary
        state.snapshot.updated_at = datetime.now(UTC)
        self._emit(
            state,
            "stage_completed",
            {
                "stage": stage,
                "status": outcome.status.value,
                "summary": outcome.result.summary,
                "new_skills": len(outcome.new_skills),
                "new_memories": len(outcome.new_memories),
            },
        )

    def _update_status(
        self,
        state: RunState,
        status: RunStatus,
        *,
        current_stage: str | None = None,
    ) -> None:
        state.status = status
        state.snapshot.status = status
        if current_stage is not None or status in (
            RunStatus.PENDING,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
        ):
            state.snapshot.current_stage = current_stage
        state.snapshot.updated_at = datetime.now(UTC)
        self._emit(state, "status_changed", {"status": status.value})

    def _fail(self, state: RunState, msg: str) -> None:
        state.snapshot.error = msg
        self._update_status(state, RunStatus.FAILED, current_stage=None)
        self._emit(state, "run_failed", {"error": msg})

    def _emit(self, state: RunState, event: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "run_id": state.run_id,
            **payload,
        }
        if self._loop is None:
            # No loop attached yet — drop the event but log it.
            _log.debug("event_dropped_no_loop", event_name=event)
            return
        asyncio.run_coroutine_threadsafe(state.queue.put(record), self._loop)

        # Mirror as a user-facing narration where applicable. We push as a
        # separate event so the UI can render technical traces and chat
        # narrations independently.
        narration = narrate_event(record)
        if narration:
            note = {
                "ts": record["ts"],
                "event": "narration",
                "run_id": state.run_id,
                "text": narration,
            }
            asyncio.run_coroutine_threadsafe(state.queue.put(note), self._loop)

    # ============================================================ subscribe

    async def subscribe(self, run_id: str) -> Iterable[dict[str, Any]]:
        state = self.get(run_id)
        if state is None:
            raise LookupError(f"Unknown run: {run_id}")
        while True:
            event = await state.queue.get()
            yield event
            if event.get("event") in {"run_completed", "run_failed"}:
                break


def _resolve_runs_dir(value: str | None):
    from pathlib import Path

    from ..config import get_settings

    if value:
        return Path(value)
    return get_settings().runs_dir
