"""Retrieval scoring per paper Eq. 12.

The Orchestrator retrieves the top-k items by a heuristic that combines:

- **keyword match**: word-level Jaccard between query and item searchable text
- **tag alignment**: explicit tag overlap (boosted vs. raw keyword)
- **recency**: exponential decay on age
- **role-specific weights**: skills add ``usage_count × confidence``;
  memories require a stricter scope/tag match before the keyword signal counts

This is intentionally lightweight (no neural embeddings) so retrieval works
without any model loaded. Embedding-based retrieval can be plugged in later
via :mod:`nanoresearch.stores.embeddings` without changing the scoring API.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

# Cheap stopword filter — we don't need NLTK for keyword scoring.
_STOPWORDS = frozenset(
    """
    a an the and or but if then of to for with on in by at from as is are be was were
    this that these those it its their our your we you they i which who what when how
    do does did doing not no yes can could should would may might must will shall about
    over under above below into onto upon off out up down here there too very also just
    while where why so than such only own same both each any all some more most other
    """.split()
)
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]+")


def tokenize(text: str) -> set[str]:
    """Lowercase tokenisation with stopword removal."""
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if len(t) > 2 and t.lower() not in _STOPWORDS
    }


class _Scorable(Protocol):
    tags: list[str]
    updated_at: datetime

    def searchable_text(self) -> str: ...  # noqa: D401, E704 - protocol


@dataclass
class RetrievalWeights:
    """Per-store weighting (paper §3.3.1 — skill vs. memory differ)."""

    keyword: float = 1.0
    tag: float = 2.0
    recency: float = 0.5
    usage: float = 0.0  # only nonzero for skills
    confidence: float = 0.0  # only nonzero for skills
    strict_scope: float = 0.0  # only nonzero for memories
    recency_half_life_days: float = 30.0


SKILL_WEIGHTS = RetrievalWeights(
    keyword=1.0,
    tag=2.0,
    recency=0.3,
    usage=0.4,
    confidence=0.8,
)
MEMORY_WEIGHTS = RetrievalWeights(
    keyword=1.0,
    tag=2.0,
    recency=0.7,
    strict_scope=1.5,
    recency_half_life_days=14.0,
)


def _recency_score(updated_at: datetime, half_life_days: float) -> float:
    delta_days = (datetime.now(UTC) - updated_at.astimezone(UTC)).total_seconds() / 86400.0
    return 0.5 ** (delta_days / max(half_life_days, 0.1))


def score(
    item: _Scorable,
    *,
    query: str,
    query_tags: set[str],
    weights: RetrievalWeights,
    require_scope: str | None = None,
    usage_count: int = 0,
    confidence: float = 0.0,
) -> float:
    """Heuristic score for one Skill/Memory against the current context."""
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    item_tokens = tokenize(item.searchable_text())
    if item_tokens:
        intersection = len(query_tokens & item_tokens)
        union = len(query_tokens | item_tokens)
        keyword = intersection / union if union else 0.0
    else:
        keyword = 0.0

    item_tags = {t.lower() for t in (item.tags or [])}
    if query_tags and item_tags:
        tag_overlap = len(query_tags & item_tags) / len(query_tags | item_tags)
    else:
        tag_overlap = 0.0

    recency = _recency_score(item.updated_at, weights.recency_half_life_days)

    s = (
        weights.keyword * keyword
        + weights.tag * tag_overlap
        + weights.recency * recency
        + weights.usage * math.log1p(usage_count)
        + weights.confidence * confidence
    )

    # Strict scope match for memories: searchable text must contain the scope
    # token(s) verbatim, otherwise the memory is heavily penalised.
    if require_scope:
        scope_tokens = tokenize(require_scope)
        if scope_tokens and not (scope_tokens & item_tokens):
            s *= 0.1  # not a hard zero — keep tiebreak signal in edge cases
        else:
            s += weights.strict_scope

    return s
