"""Skill Bank 𝒮 — JSON-backed store of compact procedural rules.

Implements:
- ``add`` / ``upsert`` — persist a new :class:`Skill`
- ``retrieve(context, k)`` — paper Eq. 12 top-k retrieval
- ``merge_overlapping(threshold)`` — paper §3.3.1 dedupe pass
- ``increment_usage`` — track usage frequency for retrieval weighting
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..schemas import Skill
from .retrieval import SKILL_WEIGHTS, score, tokenize


class SkillBank:
    """JSON-file-per-skill store with heuristic top-k retrieval."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------- CRUD

    def path_for(self, skill_id: str) -> Path:
        safe = skill_id.replace("/", "_")
        return self.root / f"{safe}.json"

    def add(self, skill: Skill) -> Skill:
        if not skill.skill_id:
            skill = skill.model_copy(update={"skill_id": f"skill-{uuid.uuid4().hex[:8]}"})
        return self.upsert(skill)

    def upsert(self, skill: Skill) -> Skill:
        skill = skill.model_copy(update={"updated_at": datetime.now(UTC)})
        self.path_for(skill.skill_id).write_text(skill.model_dump_json(indent=2))
        return skill

    def get(self, skill_id: str) -> Skill | None:
        path = self.path_for(skill_id)
        if not path.exists():
            return None
        return Skill.model_validate_json(path.read_text())

    def delete(self, skill_id: str) -> bool:
        path = self.path_for(skill_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def all(self) -> list[Skill]:
        out: list[Skill] = []
        for p in sorted(self.root.glob("*.json")):
            try:
                out.append(Skill.model_validate_json(p.read_text()))
            except (json.JSONDecodeError, ValueError):
                continue
        return out

    def __len__(self) -> int:
        return sum(1 for _ in self.root.glob("*.json"))

    # -------------------------------------------------- retrieval (Eq. 12)

    def retrieve(
        self,
        context: str,
        *,
        k: int = 5,
        tags: list[str] | None = None,
    ) -> list[Skill]:
        query_tags = {t.lower() for t in (tags or [])}
        scored: list[tuple[float, Skill]] = []
        for s in self.all():
            sc = score(
                s,
                query=context,
                query_tags=query_tags,
                weights=SKILL_WEIGHTS,
                usage_count=s.usage_count,
                confidence=s.confidence,
            )
            if sc > 0:
                scored.append((sc, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    def increment_usage(self, skill_id: str) -> Skill | None:
        s = self.get(skill_id)
        if s is None:
            return None
        s = s.model_copy(update={"usage_count": s.usage_count + 1})
        return self.upsert(s)

    # -------------------------------------------------- merge (paper §3.3.1)

    def merge_overlapping(self, *, jaccard_threshold: float = 0.7) -> int:
        """Heuristically merge skills whose searchable text Jaccard >= threshold.

        For each cluster of overlapping skills we keep the highest-confidence
        member; usage_count is summed; tags are unioned. Returns the number
        of skills removed.
        """
        skills = self.all()
        removed = 0
        consumed: set[str] = set()

        token_cache: dict[str, set[str]] = {
            s.skill_id: tokenize(s.searchable_text()) for s in skills
        }

        for i, a in enumerate(skills):
            if a.skill_id in consumed:
                continue
            cluster: list[Skill] = [a]
            for b in skills[i + 1 :]:
                if b.skill_id in consumed:
                    continue
                ta, tb = token_cache[a.skill_id], token_cache[b.skill_id]
                if not ta or not tb:
                    continue
                jacc = len(ta & tb) / len(ta | tb)
                if jacc >= jaccard_threshold:
                    consumed.add(b.skill_id)
                    cluster.append(b)

            if len(cluster) == 1:
                continue

            # Keep highest-confidence member; sum usage; union tags.
            best = max(cluster, key=lambda s: s.confidence)
            usage_sum = sum(s.usage_count for s in cluster)
            cluster_tags: set[str] = set()
            for s in cluster:
                cluster_tags.update(s.tags)

            # Remove every non-best member from disk.
            for s in cluster:
                if s.skill_id != best.skill_id:
                    self.delete(s.skill_id)
                    removed += 1

            merged = best.model_copy(
                update={
                    "tags": sorted(cluster_tags),
                    "usage_count": usage_sum,
                }
            )
            self.upsert(merged)

        return removed
