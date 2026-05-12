"""Unit tests for schemas + retrieval scoring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nanoresearch.schemas import Memory, Skill, UserProfile
from nanoresearch.stores.retrieval import (
    MEMORY_WEIGHTS,
    SKILL_WEIGHTS,
    score,
    tokenize,
)


def test_tokenize_drops_stopwords_and_short_tokens() -> None:
    toks = tokenize("The quick brown fox jumps OVER a tiny lazy dog")
    assert "fox" in toks
    assert "lazy" in toks
    # stopwords removed
    assert "the" not in toks
    assert "over" not in toks
    # short tokens removed
    assert "a" not in toks


def test_skill_searchable_text_concatenates_fields() -> None:
    s = Skill(
        skill_id="s1",
        skill_type="experiment_design_rule",
        name="One-factor ablations",
        when_to_apply="compact extension of baseline",
        procedure="vary one component at a time",
        tags=["ablation"],
    )
    text = s.searchable_text()
    assert "ablation" in text.lower()
    assert "one-factor" in text.lower() or "one factor" in text.lower()


def test_score_keyword_overlap_wins_over_unrelated() -> None:
    a = Skill(
        skill_id="a",
        name="Use saved random seeds for reproducibility",
        procedure="seed python numpy torch cuda; save splits to disk",
        tags=["reproducibility", "seed"],
    )
    b = Skill(
        skill_id="b",
        name="Write a clean abstract",
        procedure="three sentences; problem, method, result",
        tags=["writing"],
    )
    query = "How do I make the run reproducible — seeds and saved splits?"
    sa = score(a, query=query, query_tags={"reproducibility"}, weights=SKILL_WEIGHTS,
               usage_count=a.usage_count, confidence=a.confidence)
    sb = score(b, query=query, query_tags={"reproducibility"}, weights=SKILL_WEIGHTS,
               usage_count=b.usage_count, confidence=b.confidence)
    assert sa > sb


def test_score_recency_boosts_recent_items() -> None:
    old = Skill(
        skill_id="old",
        name="Reproducible seeds",
        procedure="seed everything; save splits",
        tags=["reproducibility"],
        updated_at=datetime.now(UTC) - timedelta(days=180),
    )
    new = Skill(
        skill_id="new",
        name="Reproducible seeds",
        procedure="seed everything; save splits",
        tags=["reproducibility"],
        updated_at=datetime.now(UTC),
    )
    query = "Reproducible seeds and splits"
    s_old = score(old, query=query, query_tags={"reproducibility"}, weights=SKILL_WEIGHTS,
                  usage_count=0, confidence=0)
    s_new = score(new, query=query, query_tags={"reproducibility"}, weights=SKILL_WEIGHTS,
                  usage_count=0, confidence=0)
    assert s_new > s_old


def test_score_skill_usage_and_confidence_boost() -> None:
    base = Skill(
        skill_id="b",
        name="One-factor ablations",
        procedure="vary one component at a time",
        tags=["ablation"],
    )
    promoted = base.model_copy(update={"skill_id": "p", "usage_count": 25, "confidence": 0.95})
    q = "design ablations that isolate one factor"
    s_base = score(base, query=q, query_tags={"ablation"}, weights=SKILL_WEIGHTS,
                   usage_count=base.usage_count, confidence=base.confidence)
    s_promoted = score(promoted, query=q, query_tags={"ablation"}, weights=SKILL_WEIGHTS,
                       usage_count=promoted.usage_count, confidence=promoted.confidence)
    assert s_promoted > s_base


def test_score_memory_strict_scope_penalises_mismatch() -> None:
    m = Memory(
        memory_id="m1",
        topic_scope="UCI HAR sensor classification",
        content="Use 1D CNN + GRU + InceptionTime-small as baseline suite",
        tags=["uci_har"],
    )
    matching = score(
        m,
        query="UCI HAR baseline suite for sensor classification",
        query_tags={"uci_har"},
        weights=MEMORY_WEIGHTS,
        require_scope="UCI HAR sensor classification",
    )
    mismatched = score(
        m,
        query="ImageNet classification baselines",
        query_tags={"imagenet"},
        weights=MEMORY_WEIGHTS,
        require_scope="ImageNet classification",
    )
    assert matching > 0
    assert mismatched < matching * 0.5


def test_userprofile_roundtrip() -> None:
    p = UserProfile(
        user_id="alice",
        archetype="ai4science_journal",
        domain="Time Series",
        risk_preference="low",  # type: ignore[arg-type]
        baseline_strictness="very_high",  # type: ignore[arg-type]
    )
    blob = p.model_dump_json()
    restored = UserProfile.model_validate_json(blob)
    assert restored.archetype == "ai4science_journal"
    assert restored.user_id == "alice"
