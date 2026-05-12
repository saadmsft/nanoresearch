"""Unit tests for the run manifest."""

from __future__ import annotations

import json
from pathlib import Path

from nanoresearch.logging import RunManifest, configure_logging


def test_manifest_records_events(tmp_path: Path) -> None:
    configure_logging("INFO")
    m = RunManifest(tmp_path / "runs")
    m.record("started", topic="UCI HAR")
    m.record_llm_call(
        role="planner",
        backend="local-qwen",
        model="Qwen2.5-7B-Instruct",
        prompt_tokens=10,
        completion_tokens=20,
        latency_ms=123.4,
        prompt_preview="What is the plan?",
        completion_preview="The plan is X",
    )
    m.stage("ideation", "completed")

    lines = m.events_path.read_text().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]

    assert records[0]["event"] == "started"
    assert records[0]["topic"] == "UCI HAR"

    assert records[1]["event"] == "llm_call"
    assert records[1]["role"] == "planner"
    assert records[1]["latency_ms"] == 123.4

    assert records[2]["event"] == "stage"
    assert records[2]["stage"] == "ideation"


def test_manifest_truncates_long_previews(tmp_path: Path) -> None:
    m = RunManifest(tmp_path / "runs")
    long_text = "x" * 5000
    m.record_llm_call(
        role="writing",
        backend="azure-foundry",
        model="gpt-5.1",
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=0.0,
        prompt_preview=long_text,
        completion_preview=long_text,
    )
    record = json.loads(m.events_path.read_text())
    assert len(record["prompt_preview"]) <= 500
    assert record["prompt_preview"].endswith("…")
