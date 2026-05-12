"""Shared helpers for stage agents."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Tolerant extraction of a single JSON object from an LLM response.

    Order of attempts: fenced block → bare text → outermost ``{ … }`` slice.
    Returns ``None`` if no valid JSON object is found.
    """
    if not text:
        return None
    candidates: list[str] = []
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    candidates.append(text.strip())
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])
    for c in candidates:
        try:
            data = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def render_skills(skills: list[Any], *, max_items: int = 5) -> str:
    """Render the top retrieved skills as a compact bulleted list for prompts."""
    if not skills:
        return "(no skills retrieved)"
    lines: list[str] = []
    for s in skills[:max_items]:
        lines.append(f"- **{s.name}**: {s.procedure[:200]}")
        if s.do_not:
            lines.append(f"    DO NOT: {s.do_not[:160]}")
    return "\n".join(lines)


def render_memories(memories: list[Any], *, max_items: int = 5) -> str:
    if not memories:
        return "(no memories retrieved)"
    lines: list[str] = []
    for m in memories[:max_items]:
        lines.append(f"- ({m.memory_type}) scope='{m.topic_scope}': {m.content[:240]}")
        if m.failure_mode_to_avoid:
            lines.append(f"    avoid: {m.failure_mode_to_avoid[:160]}")
    return "\n".join(lines)


def render_papers(papers: list[Any], *, max_items: int = 10) -> str:
    if not papers:
        return "(no papers retrieved)"
    lines: list[str] = []
    for p in papers[:max_items]:
        cite = p.short_citation() if hasattr(p, "short_citation") else str(p)
        lines.append(f"- [{p.paper_id}] {cite}  cites={p.citations}")
        if p.abstract:
            lines.append(f"    {p.abstract[:280]}")
    return "\n".join(lines)


def render_evidence(evs: list[Any], *, max_items: int = 8) -> str:
    if not evs:
        return "(no quantitative evidence extracted)"
    lines: list[str] = []
    for e in evs[:max_items]:
        lines.append(f"- {e.metric}={e.value}  from='{e.method[:80]}'  ctx='{e.snippet[:120]}'")
    return "\n".join(lines)
