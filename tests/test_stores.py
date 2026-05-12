"""Unit tests for SkillBank, MemoryStore, and ProfileStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from nanoresearch.schemas import Memory, Skill, UserProfile
from nanoresearch.stores import MemoryStore, ProfileStore, SkillBank


def _make_skill(name: str, **kw: object) -> Skill:
    return Skill(skill_id="", name=name, procedure=name + " procedure", **kw)  # type: ignore[arg-type]


def test_skill_bank_add_get_retrieve(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path)

    a = bank.add(_make_skill("Reproducible seeds and saved splits", tags=["reproducibility"]))
    b = bank.add(_make_skill("Write a clean abstract", tags=["writing"]))

    assert len(bank) == 2
    assert bank.get(a.skill_id) is not None
    assert bank.get(b.skill_id) is not None

    top = bank.retrieve("Make my training run reproducible", tags=["reproducibility"], k=1)
    assert len(top) == 1
    assert top[0].skill_id == a.skill_id


def test_skill_bank_usage_counter_persists(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path)
    s = bank.add(_make_skill("debug rule"))
    bank.increment_usage(s.skill_id)
    bank.increment_usage(s.skill_id)
    fresh = bank.get(s.skill_id)
    assert fresh is not None
    assert fresh.usage_count == 2


def test_skill_bank_merge_dedupes_similar(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path)
    bank.add(
        Skill(
            skill_id="s1",
            name="Design one-factor ablations",
            procedure="vary one component at a time keeping data optimizer schedule fixed",
            tags=["ablation"],
            usage_count=3,
            confidence=0.6,
        )
    )
    bank.add(
        Skill(
            skill_id="s2",
            name="Design one-factor ablations",
            procedure="vary one component at a time keeping data optimizer schedule fixed and metric",
            tags=["ablation", "design"],
            usage_count=2,
            confidence=0.8,
        )
    )

    removed = bank.merge_overlapping(jaccard_threshold=0.7)
    assert removed == 1
    assert len(bank) == 1
    kept = bank.all()[0]
    # higher-confidence kept
    assert kept.confidence == 0.8
    # usage counts summed; tags unioned
    assert kept.usage_count == 5
    assert set(kept.tags) == {"ablation", "design"}


def test_memory_store_strict_scope(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add(
        Memory(
            memory_id="m_har",
            topic_scope="UCI HAR sensor classification",
            content="Baseline suite: 1D CNN, GRU, InceptionTime-small",
            tags=["uci_har"],
        )
    )
    store.add(
        Memory(
            memory_id="m_other",
            topic_scope="ImageNet classification",
            content="Baseline suite: ResNet-50, ViT-B/16",
            tags=["imagenet"],
        )
    )

    har_hits = store.retrieve(
        "lightweight UCI HAR baselines",
        tags=["uci_har"],
        topic_scope="UCI HAR sensor classification",
        k=5,
    )
    assert [m.memory_id for m in har_hits][0] == "m_har"


def test_memory_store_merge_only_within_scope(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.add(
        Memory(
            memory_id="a",
            topic_scope="UCI HAR sensor classification",
            content="Use same train/val/test split across baselines and proposed method.",
            tags=["uci_har"],
        )
    )
    store.add(
        Memory(
            memory_id="b",
            topic_scope="UCI HAR sensor classification",
            content="Use same train/val/test split across baselines and proposed method.",
            tags=["uci_har", "splits"],
        )
    )
    # Different scope, same content shouldn't be merged
    store.add(
        Memory(
            memory_id="c",
            topic_scope="ImageNet",
            content="Use same train/val/test split across baselines and proposed method.",
            tags=["imagenet"],
        )
    )
    removed = store.merge_overlapping(jaccard_threshold=0.8)
    assert removed == 1  # a + b merged, c left alone
    assert len(store) == 2


def test_profile_store_roundtrip(tmp_path: Path) -> None:
    store = ProfileStore(tmp_path)
    profile = UserProfile(
        user_id="alice",
        archetype="ai4science_journal",
        domain="Time Series",
    )
    store.save(profile)
    assert store.exists("alice")
    loaded = store.load("alice")
    assert loaded is not None
    assert loaded.archetype == "ai4science_journal"

    # Standard subdirs exist
    assert store.skills_dir("alice").exists()
    assert store.memories_dir("alice").exists()
    assert store.lora_dir("alice").exists()
    assert "alice" in store.list_users()


def test_skill_bank_skips_corrupt_files(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path)
    bank.add(_make_skill("good"))
    (tmp_path / "broken.json").write_text("{ not valid json")
    skills = bank.all()
    assert len(skills) == 1


def test_skill_recency_decay(tmp_path: Path) -> None:
    bank = SkillBank(tmp_path)
    old = Skill(
        skill_id="old",
        name="Reproducible seeds",
        procedure="seed everything",
        tags=["reproducibility"],
        updated_at=datetime.now(UTC) - timedelta(days=400),
    )
    new = Skill(
        skill_id="new",
        name="Reproducible seeds",
        procedure="seed everything",
        tags=["reproducibility"],
        updated_at=datetime.now(UTC),
    )
    # Directly upsert to preserve our backdated timestamp on `old`
    bank.path_for("old").write_text(old.model_dump_json())
    bank.path_for("new").write_text(new.model_dump_json())

    results = bank.retrieve("Reproducible seeds", tags=["reproducibility"], k=2)
    assert [s.skill_id for s in results] == ["new", "old"]
