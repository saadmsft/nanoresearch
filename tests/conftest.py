"""Pytest fixtures shared across the suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nanoresearch.config import reset_settings_cache


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a sane default environment for every test.

    Tests that need real Azure creds should set the env vars in their own
    fixture and call ``reset_settings_cache()``.
    """
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        os.getenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/"),
    )
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LORA_ADAPTERS_DIR", str(tmp_path / "data" / "users"))
    reset_settings_cache()
    yield
    reset_settings_cache()
