"""Unit tests for the distillation pipeline (paper Eq. 13)."""

from __future__ import annotations

import json
from typing import Any

from nanoresearch.llm import (
    AgentRole,
    ChatMessage,
    CompletionResult,
    LLMBackend,
    LLMRouter,
)
from nanoresearch.stores.distill import _extract_json, distill


class _ScriptedBackend(LLMBackend):
    name = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(  # type: ignore[override]
        self,
        messages: list[ChatMessage],
        **_: Any,
    ) -> CompletionResult:
        return CompletionResult(
            text=self.reply,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            finish_reason="stop",
            backend=self.name,
            model="scripted",
            latency_ms=1.0,
        )


def test_extract_json_handles_fenced_and_bare() -> None:
    assert _extract_json('{"a": 1}') == {"a": 1}
    assert _extract_json('Sure! ```json\n{"a": 2}\n```') == {"a": 2}
    assert _extract_json("noise {\"a\": 3} more noise") == {"a": 3}
    assert _extract_json("not json at all") is None
    assert _extract_json("") is None


def test_distill_produces_validated_skill_and_memory() -> None:
    reply = json.dumps(
        {
            "skills": [
                {
                    "skill_type": "experiment_design_rule",
                    "name": "Design one-factor ablations",
                    "when_to_apply": "Compact extension of baseline.",
                    "procedure": "Vary one component at a time keeping training fixed.",
                    "planning_effect": "Blueprint lists ablation groups per claim.",
                    "coding_effect": "Use variant flags in one constructor.",
                    "writing_effect": "",
                    "analysis_effect": "Attribute gains to one component.",
                    "review_check": "Reviewer can isolate the contribution.",
                    "do_not": "Bundle multiple changes per ablation.",
                    "tags": ["ablation", "design"],
                }
            ],
            "memories": [
                {
                    "memory_type": "project_context",
                    "source_stage": "planning",
                    "topic_scope": "UCI HAR sensor classification",
                    "content": "Baseline suite: 1D CNN, GRU, InceptionTime-small with shared split.",
                    "retrieval_rationale": "Same UCI HAR setup.",
                    "planning_implication": "Reproduce baselines before proposing new methods.",
                    "coding_implication": "Single shared dataloader.",
                    "analysis_implication": "Report metrics on the shared split only.",
                    "writing_implication": "Frame the contribution as a controlled extension.",
                    "failure_mode_to_avoid": "Different schedules per baseline.",
                    "tags": ["uci_har", "baselines"],
                }
            ],
        }
    )
    router = LLMRouter(azure=_ScriptedBackend(reply), planner=_ScriptedBackend(""))
    out = distill(
        router=router,
        trajectory_summary="ideation produced UCI HAR plan with three baselines",
        stage="ideation",
        user_id="alice",
        project_id="uci-har-1",
    )
    assert len(out.skills) == 1
    assert out.skills[0].name == "Design one-factor ablations"
    assert out.skills[0].skill_id.startswith("skill-")
    assert len(out.memories) == 1
    assert out.memories[0].topic_scope == "UCI HAR sensor classification"
    assert out.memories[0].user_id == "alice"
    assert out.memories[0].project_id == "uci-har-1"


def test_distill_returns_empty_on_garbage() -> None:
    router = LLMRouter(azure=_ScriptedBackend("totally not JSON"), planner=_ScriptedBackend(""))
    out = distill(router=router, trajectory_summary="x", stage="ideation")
    assert out.skills == []
    assert out.memories == []


def test_distill_skips_invalid_entries_but_keeps_valid() -> None:
    reply = json.dumps(
        {
            "skills": [
                {"name": "Missing required fields"},  # missing skill_type? type is "other" default
            ],
            "memories": [
                {"memory_type": "project_context"},  # mostly empty, but valid (all optional strings)
                "not-a-dict",  # outright invalid
            ],
        }
    )
    router = LLMRouter(azure=_ScriptedBackend(reply), planner=_ScriptedBackend(""))
    out = distill(router=router, trajectory_summary="x", stage="ideation")
    # The "Missing required fields" skill has name => valid; default skill_type=other
    assert len(out.skills) == 1
    assert out.skills[0].name == "Missing required fields"
    # The "memory_type" entry is valid (all other memory fields are optional defaults to "")
    assert len(out.memories) == 1
    assert out.memories[0].memory_type == "project_context"
