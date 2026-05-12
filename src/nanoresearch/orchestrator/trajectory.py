"""In-memory trajectory ``τ`` — actions, critiques, outcomes for distillation.

After each stage the Orchestrator passes the trajectory's textual summary to
:func:`nanoresearch.stores.distill.distill` (paper Eq. 13) to produce new
Skills and Memories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class EventKind(StrEnum):
    ACTION = "action"
    PROMPT = "prompt"
    RESPONSE = "response"
    CRITIQUE = "critique"
    OUTCOME = "outcome"
    ERROR = "error"
    NOTE = "note"


@dataclass
class TrajectoryEvent:
    kind: EventKind
    label: str
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Trajectory:
    """Append-only record for one stage of one run."""

    stage: str
    events: list[TrajectoryEvent] = field(default_factory=list)

    def add(
        self,
        kind: EventKind,
        label: str,
        detail: str = "",
        **metadata: Any,
    ) -> TrajectoryEvent:
        ev = TrajectoryEvent(kind=kind, label=label, detail=detail, metadata=metadata)
        self.events.append(ev)
        return ev

    # ---- convenience helpers --------------------------------------------
    def action(self, label: str, detail: str = "", **meta: Any) -> TrajectoryEvent:
        return self.add(EventKind.ACTION, label, detail, **meta)

    def critique(self, label: str, detail: str = "", **meta: Any) -> TrajectoryEvent:
        return self.add(EventKind.CRITIQUE, label, detail, **meta)

    def outcome(self, label: str, detail: str = "", **meta: Any) -> TrajectoryEvent:
        return self.add(EventKind.OUTCOME, label, detail, **meta)

    def error(self, label: str, detail: str = "", **meta: Any) -> TrajectoryEvent:
        return self.add(EventKind.ERROR, label, detail, **meta)

    # ---- distillation input ---------------------------------------------
    def summarise(self, *, max_chars: int = 6000) -> str:
        """Compact textual representation used by the distiller LLM."""
        lines = [f"# Stage: {self.stage}", ""]
        for e in self.events:
            head = f"[{e.kind.value.upper()}] {e.label}"
            lines.append(head)
            if e.detail:
                lines.append(_truncate(e.detail, 600))
            if e.metadata:
                meta_str = "; ".join(
                    f"{k}={_short(v)}" for k, v in e.metadata.items() if v is not None
                )
                if meta_str:
                    lines.append(f"  ({meta_str})")
            lines.append("")
        text = "\n".join(lines).strip()
        if len(text) > max_chars:
            text = text[: max_chars - 20] + "\n…[truncated]"
        return text


def _truncate(text: str, n: int) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _short(value: Any) -> str:
    s = str(value)
    if len(s) > 80:
        return s[:79] + "…"
    return s
