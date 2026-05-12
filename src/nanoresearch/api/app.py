"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ..config import get_settings
from ..llm import LLMRouter
from ..logging import configure_logging, get_logger
from ..schemas import UserProfile
from ..stores import MemoryStore, ProfileStore, SkillBank
from .intent import IntentRequest, ParsedIntent, parse_intent
from .run_manager import RunManager, RunSnapshot, RunStatus

_log = get_logger("api")


# ============================================================ DTOs


class UserProfileCreate(BaseModel):
    user_id: str
    archetype: str
    domain: str
    research_preference: str = ""
    method_preference: str = ""
    risk_preference: str = "moderate"
    baseline_strictness: str = "high"
    resource_budget: str = ""
    feasibility_bias: str = ""
    writing_tone: str = ""
    claim_strength: str = ""
    section_organization: str = ""
    venue_style: str = ""
    latex_template: str = "conference_template"
    persona_brief: str = ""


class StartRunRequest(BaseModel):
    user_id: str
    topic: str = Field(..., min_length=4)


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1)


# ============================================================ factory


def create_app(
    *,
    router: LLMRouter | None = None,
    profile_store: ProfileStore | None = None,
    run_manager: RunManager | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Each dependency is optional so tests can inject fakes. Production
    callers should rely on the defaults — they wire the real
    :class:`LLMRouter` and :class:`ProfileStore`.
    """
    configure_logging(get_settings().log_level)

    settings = get_settings()
    _router = router or LLMRouter()
    _profile_store = profile_store or ProfileStore(settings.lora_adapters_dir)
    _runs = run_manager or RunManager(
        router=_router,
        profile_store=_profile_store,
        runs_dir=str(settings.runs_dir),
    )

    app = FastAPI(
        title="NanoResearch API",
        version="0.1.0",
        description="HTTP interface for the NanoResearch tri-level co-evolution pipeline.",
    )

    # Local UI dev server runs on a different port — allow CORS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------- DI
    def get_profile_store() -> ProfileStore:
        return _profile_store

    def get_run_manager() -> RunManager:
        return _runs

    @app.on_event("startup")
    async def _attach_loop() -> None:
        _runs.attach_loop(asyncio.get_running_loop())

    # ------------------------------------------------------------- health
    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": app.version}

    # ------------------------------------------------------------- users
    @app.post("/api/users", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
    def upsert_user(
        body: UserProfileCreate, store: ProfileStore = Depends(get_profile_store)
    ) -> UserProfile:
        profile = UserProfile(**body.model_dump())
        return store.save(profile)

    @app.get("/api/users", response_model=list[str])
    def list_users(store: ProfileStore = Depends(get_profile_store)) -> list[str]:
        return store.list_users()

    @app.get("/api/users/{user_id}", response_model=UserProfile)
    def get_user(user_id: str, store: ProfileStore = Depends(get_profile_store)) -> UserProfile:
        profile = store.load(user_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        return profile

    @app.get("/api/users/{user_id}/skills")
    def get_skills(user_id: str, store: ProfileStore = Depends(get_profile_store)) -> list[dict[str, Any]]:
        if not store.exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        sb = SkillBank(store.skills_dir(user_id))
        return [s.model_dump(mode="json") for s in sb.all()]

    @app.get("/api/users/{user_id}/memories")
    def get_memories(user_id: str, store: ProfileStore = Depends(get_profile_store)) -> list[dict[str, Any]]:
        if not store.exists(user_id):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        ms = MemoryStore(store.memories_dir(user_id))
        return [m.model_dump(mode="json") for m in ms.all()]

    # ------------------------------------------------------------- intent
    @app.post("/api/intent")
    def post_intent(body: IntentRequest) -> dict[str, Any]:
        """Classify a free-form user message into an :class:`Intent`."""
        parsed: ParsedIntent = parse_intent(
            router=_router, text=body.text, session=body.session
        )
        return {
            "source": parsed.source,
            "intent": parsed.intent.model_dump(),
        }

    # ------------------------------------------------------------- runs
    @app.post("/api/runs", response_model=RunSnapshot, status_code=status.HTTP_201_CREATED)
    def start_run(body: StartRunRequest, runs: RunManager = Depends(get_run_manager)) -> RunSnapshot:
        try:
            return runs.start_run(user_id=body.user_id, topic=body.topic)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/api/runs", response_model=list[RunSnapshot])
    def list_runs(runs: RunManager = Depends(get_run_manager)) -> list[RunSnapshot]:
        return runs.list()

    @app.get("/api/runs/{run_id}", response_model=RunSnapshot)
    def get_run(run_id: str, runs: RunManager = Depends(get_run_manager)) -> RunSnapshot:
        state = runs.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        return state.snapshot

    @app.post("/api/runs/{run_id}/feedback", response_model=RunSnapshot)
    def post_feedback(
        run_id: str, body: FeedbackRequest, runs: RunManager = Depends(get_run_manager)
    ) -> RunSnapshot:
        try:
            return runs.submit_feedback(run_id, body.text)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

    @app.get("/api/runs/{run_id}/stream")
    async def stream(run_id: str, runs: RunManager = Depends(get_run_manager)):
        state = runs.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

        async def event_generator():
            async for ev in runs.subscribe(run_id):
                yield {
                    "event": ev.get("event", "message"),
                    "data": json.dumps(ev, default=str),
                }
            yield {"event": "stream_end", "data": "{}"}

        return EventSourceResponse(event_generator())

    # ------------------------------------------------------------- artefacts
    @app.get("/api/runs/{run_id}/paper.pdf")
    def paper_pdf(run_id: str, runs: RunManager = Depends(get_run_manager)) -> FileResponse:
        state = runs.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        compiled = state.artefacts.get("compiled")
        if compiled is None or not compiled.compiled or not compiled.pdf_path:
            raise HTTPException(status_code=404, detail="PDF not available for this run.")
        return FileResponse(
            compiled.pdf_path,
            media_type="application/pdf",
            filename=f"{run_id}.pdf",
        )

    @app.get("/api/runs/{run_id}/paper.tex")
    def paper_tex(run_id: str, runs: RunManager = Depends(get_run_manager)) -> FileResponse:
        state = runs.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        compiled = state.artefacts.get("compiled")
        if compiled is None or not compiled.tex_path:
            raise HTTPException(status_code=404, detail="LaTeX source not available for this run.")
        return FileResponse(
            compiled.tex_path,
            media_type="text/x-tex",
            filename=f"{run_id}.tex",
        )

    return app


# Expose a module-level default app so `uvicorn nanoresearch.api.app:app` works.
app = create_app()
