"""Structured logging + per-run manifest for reproducibility.

The run manifest records every LLM call (prompts, responses, latency, token
counts, identity) so that successful trajectories can be distilled into the
Skill Bank and Memory Module (paper §3.3).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog + stdlib logging once at process start."""
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO),
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


class RunManifest:
    """Append-only JSONL log for one NanoResearch run.

    Records every LLM call + stage transition. The orchestrator distills
    skills/memories from this trajectory at the end of each stage.
    """

    def __init__(self, runs_dir: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or _make_run_id()
        self.run_dir = runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self._logger = get_logger("manifest").bind(run_id=self.run_id)

    def record(self, event_type: str, **payload: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event_type,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")
        self._logger.debug("manifest_event", event_type=event_type)

    def record_llm_call(
        self,
        *,
        role: str,
        backend: str,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_ms: float,
        prompt_preview: str,
        completion_preview: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.record(
            "llm_call",
            role=role,
            backend=backend,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 2),
            prompt_preview=_truncate(prompt_preview, 500),
            completion_preview=_truncate(completion_preview, 500),
            extra=extra or {},
        )

    def stage(self, stage: str, status: str, **payload: Any) -> None:
        self.record("stage", stage=stage, status=status, **payload)


def _make_run_id() -> str:
    return f"run-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _truncate(text: str, n: int) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Unserialisable type: {type(obj).__name__}")
