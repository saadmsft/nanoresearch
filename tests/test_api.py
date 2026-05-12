"""HTTP API tests.

We inject scripted backends + a temp ProfileStore so the API can be exercised
end-to-end without Azure or local model weights.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nanoresearch.api.app import create_app
from nanoresearch.api.run_manager import RunManager, RunStatus
from nanoresearch.llm import (
    ChatMessage,
    CompletionResult,
    LLMBackend,
    LLMRouter,
)
from nanoresearch.stores import ProfileStore


class ScriptedBackend(LLMBackend):
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage], **_: Any) -> CompletionResult:
        self.calls.append(messages)
        text = self.replies.pop(0) if self.replies else "{}"
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


@pytest.fixture
def client_and_runs(tmp_path: Path) -> tuple[TestClient, RunManager, ProfileStore]:
    profile_store = ProfileStore(tmp_path / "users")
    backend = ScriptedBackend(replies=["{}"] * 50)  # generous so we never run out
    router = LLMRouter(azure=backend, planner=ScriptedBackend([]))
    runs = RunManager(router=router, profile_store=profile_store, runs_dir=str(tmp_path / "runs"))
    app = create_app(router=router, profile_store=profile_store, run_manager=runs)
    client = TestClient(app)
    return client, runs, profile_store


# ============================================================ users


def test_create_and_fetch_user(client_and_runs) -> None:
    client, _, _ = client_and_runs
    payload = {"user_id": "alice", "archetype": "ai4science_journal", "domain": "Time Series"}
    res = client.post("/api/users", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["user_id"] == "alice"
    assert data["archetype"] == "ai4science_journal"

    fetched = client.get("/api/users/alice")
    assert fetched.status_code == 200
    assert fetched.json()["user_id"] == "alice"


def test_get_user_404(client_and_runs) -> None:
    client, _, _ = client_and_runs
    res = client.get("/api/users/nobody")
    assert res.status_code == 404


def test_list_users(client_and_runs) -> None:
    client, _, _ = client_and_runs
    client.post("/api/users", json={"user_id": "alice", "archetype": "x", "domain": "NLP"})
    client.post("/api/users", json={"user_id": "bob", "archetype": "y", "domain": "CV"})
    res = client.get("/api/users")
    assert res.status_code == 200
    assert set(res.json()) == {"alice", "bob"}


def test_skills_and_memories_empty(client_and_runs) -> None:
    client, _, _ = client_and_runs
    client.post("/api/users", json={"user_id": "alice", "archetype": "x", "domain": "NLP"})
    res = client.get("/api/users/alice/skills")
    assert res.status_code == 200
    assert res.json() == []
    res = client.get("/api/users/alice/memories")
    assert res.status_code == 200
    assert res.json() == []


# ============================================================ runs


def test_start_run_unknown_user_returns_404(client_and_runs) -> None:
    client, _, _ = client_and_runs
    res = client.post("/api/runs", json={"user_id": "ghost", "topic": "anything"})
    assert res.status_code == 404


def test_start_run_returns_pending_snapshot(client_and_runs) -> None:
    client, _, _ = client_and_runs
    client.post("/api/users", json={"user_id": "alice", "archetype": "x", "domain": "NLP"})
    res = client.post(
        "/api/runs",
        json={"user_id": "alice", "topic": "Lightweight PubMedQA fine-tuning"},
    )
    assert res.status_code == 201
    snap = res.json()
    assert snap["user_id"] == "alice"
    assert snap["topic"] == "Lightweight PubMedQA fine-tuning"
    assert snap["status"] in {"pending", "running", "awaiting_feedback"}


def test_feedback_409_when_not_awaiting(client_and_runs) -> None:
    client, runs, store = client_and_runs
    client.post("/api/users", json={"user_id": "alice", "archetype": "x", "domain": "NLP"})
    snap = client.post("/api/runs", json={"user_id": "alice", "topic": "any topic here"}).json()
    run_id = snap["run_id"]
    # Force state to running for the test:
    state = runs.get(run_id)
    assert state is not None
    state.status = RunStatus.RUNNING
    state.snapshot.status = RunStatus.RUNNING

    res = client.post(f"/api/runs/{run_id}/feedback", json={"text": "more terse"})
    assert res.status_code == 409
