"""End-to-end Azure smoke test.

Marked ``azure`` — opt-in via ``pytest -m azure``. Requires:
- ``az login`` performed (DefaultAzureCredential)
- Role assignment: "Cognitive Services OpenAI User" on the resource
- ``AZURE_OPENAI_ENDPOINT`` + ``AZURE_OPENAI_DEPLOYMENT`` set
"""

from __future__ import annotations

import os

import pytest

from nanoresearch.llm import AzureFoundryClient, ChatMessage, Role


@pytest.mark.azure
def test_azure_foundry_round_trip() -> None:
    if "AZURE_OPENAI_ENDPOINT" not in os.environ:
        pytest.skip("AZURE_OPENAI_ENDPOINT not set")

    client = AzureFoundryClient()
    result = client.complete(
        [
            ChatMessage(Role.SYSTEM, "You are a terse echo. Reply with exactly: OK"),
            ChatMessage(Role.USER, "Reply OK"),
        ],
        max_tokens=8,
        temperature=0.0,
    )
    assert "OK" in result.text.upper()
    assert result.backend == "azure-foundry"
    assert result.prompt_tokens is None or result.prompt_tokens > 0
