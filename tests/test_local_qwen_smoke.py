"""Local-model smoke test (heavy).

Marked ``local_model`` + ``slow`` — opt-in via ``pytest -m local_model``.
Requires the ``[local]`` extras and Qwen2.5-7B weights on disk.
"""

from __future__ import annotations

import importlib.util

import pytest

from nanoresearch.llm import AgentRole, ChatMessage, LLMRouter, Role


@pytest.mark.local_model
@pytest.mark.slow
def test_qwen_planner_round_trip() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed (install [local] extras)")
    if importlib.util.find_spec("transformers") is None:
        pytest.skip("transformers not installed (install [local] extras)")

    router = LLMRouter()
    result = router.complete(
        AgentRole.PLANNER,
        [ChatMessage(Role.USER, "Reply with exactly: OK")],
        max_tokens=8,
        temperature=0.0,
    )
    assert "OK" in result.text.upper()
    assert result.backend == "local-qwen"
