"""Unit tests for the LLM router.

These tests use a fake backend so they run without Azure creds or torch.
The real Azure round-trip is exercised by ``test_azure_smoke.py`` under the
``azure`` marker.
"""

from __future__ import annotations

from typing import Any

import pytest

from nanoresearch.llm import (
    AgentRole,
    ChatMessage,
    CompletionResult,
    LLMBackend,
    LLMRouter,
    Role,
)
from nanoresearch.logging import RunManifest


class _FakeBackend(LLMBackend):
    name = "fake"

    def __init__(self, reply: str = "OK") -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CompletionResult:
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return CompletionResult(
            text=self.reply,
            prompt_tokens=sum(len(m.content.split()) for m in messages),
            completion_tokens=len(self.reply.split()),
            total_tokens=None,
            finish_reason="stop",
            backend=self.name,
            model="fake-model",
            latency_ms=1.0,
        )


def test_router_routes_planner_to_local_backend() -> None:
    azure = _FakeBackend("from-azure")
    planner = _FakeBackend("from-planner")
    router = LLMRouter(azure=azure, planner=planner)

    msgs = [ChatMessage(Role.USER, "hi")]
    plan = router.complete(AgentRole.PLANNER, msgs)
    assert plan.text == "from-planner"
    assert len(planner.calls) == 1
    assert len(azure.calls) == 0


def test_router_routes_other_roles_to_azure() -> None:
    azure = _FakeBackend("from-azure")
    planner = _FakeBackend("from-planner")
    router = LLMRouter(azure=azure, planner=planner)

    msgs = [ChatMessage(Role.USER, "hi")]
    for role in (
        AgentRole.IDEATION,
        AgentRole.PLANNING,
        AgentRole.CODING,
        AgentRole.WRITING,
        AgentRole.REVIEW,
        AgentRole.JUDGE_COMPLIANCE,
    ):
        res = router.complete(role, msgs)
        assert res.text == "from-azure"
    assert len(azure.calls) == 6
    assert len(planner.calls) == 0


def test_router_records_calls_to_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    azure = _FakeBackend()
    planner = _FakeBackend()
    manifest = RunManifest(tmp_path / "runs")
    router = LLMRouter(azure=azure, planner=planner, manifest=manifest)

    router.complete(AgentRole.WRITING, [ChatMessage(Role.USER, "write")])
    lines = manifest.events_path.read_text().splitlines()
    assert len(lines) == 1
    assert '"role": "writing"' in lines[0]


def test_router_rejects_unmapped_role() -> None:
    router = LLMRouter(azure=_FakeBackend(), planner=_FakeBackend())
    with pytest.raises(ValueError):
        router.backend_for("not-a-role")  # type: ignore[arg-type]
