"""Unit tests for :mod:`nanoresearch.config`."""

from __future__ import annotations

import pytest

from nanoresearch.config import Device, get_settings, reset_settings_cache


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://foo.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")
    reset_settings_cache()

    s = get_settings()
    assert s.azure_openai_deployment == "gpt-5.1"
    assert s.azure_endpoint_str == "https://foo.openai.azure.com"
    assert s.local_model_device is Device.MPS


def test_blank_deployment_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://foo.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "  ")
    reset_settings_cache()
    with pytest.raises(Exception):  # pydantic-settings wraps ValidationError
        get_settings()


def test_endpoint_trailing_slash_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1")
    reset_settings_cache()
    assert not get_settings().azure_endpoint_str.endswith("/")
