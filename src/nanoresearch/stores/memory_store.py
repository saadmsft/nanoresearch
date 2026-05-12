"""Memory Module ℳ — JSON-backed store of project-bound experience.

Differs from :class:`SkillBank` in two ways:

- Retrieval enforces a **strict scope match** (paper §3.3.1) — a memory only
  surfaces when its scope tokens overlap with the current context.
- No usage/confidence weighting: memories are facts, not strategies.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from ..schemas import Memory
from .retrieval import MEMORY_WEIGHTS, score, tokenize


class MemoryStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------- CRUD

    def path_for(self, memory_id: str) -> Path:
        safe = memory_id.replace("/", "_")
        return self.root / f"{safe}.json"

    def add(self, memory: Memory) -> Memory:
        if not memory.memory_id:
            memory = memory.model_copy(update={"memory_id": f"mem-{uuid.uuid4().hex[:8]}"})
        return self.upsert(memory)

    def upsert(self, memory: Memory) -> Memory:
        memory = memory.model_copy(update={"updated_at": datetime.now(UTC)})
        self.path_for(memory.memory_id).write_text(memory.model_dump_json(indent=2))
        return memory

    def get(self, memory_id: str) -> Memory | None:
        path = self.path_for(memory_id)
        if not path.exists():
            return None
        return Memory.model_validate_json(path.read_text())

    def delete(self, memory_id: str) -> bool:
        path = self.path_for(memory_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def all(self) -> list[Memory]:
        out: list[Memory] = []
        for p in sorted(self.root.glob("*.json")):
            try:
                out.append(Memory.model_validate_json(p.read_text()))
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
        topic_scope: str | None = None,
    ) -> list[Memory]:
        query_tags = {t.lower() for t in (tags or [])}
        scored: list[tuple[float, Memory]] = []
        for m in self.all():
            sc = score(
                m,
                query=context,
                query_tags=query_tags,
                weights=MEMORY_WEIGHTS,
                require_scope=topic_scope or m.topic_scope or None,
            )
            if sc > 0:
                scored.append((sc, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:k]]

    # -------------------------------------------------- merge

    def merge_overlapping(self, *, jaccard_threshold: float = 0.8) -> int:
        """Higher merge threshold than skills — memories are facts, harder to merge."""
        memories = self.all()
        removed = 0
        consumed: set[str] = set()
        token_cache = {m.memory_id: tokenize(m.searchable_text()) for m in memories}

        for i, a in enumerate(memories):
            if a.memory_id in consumed:
                continue
            cluster: list[Memory] = [a]
            for b in memories[i + 1 :]:
                if b.memory_id in consumed:
                    continue
                if a.topic_scope and b.topic_scope and a.topic_scope != b.topic_scope:
                    continue
                ta, tb = token_cache[a.memory_id], token_cache[b.memory_id]
                if not ta or not tb:
                    continue
                jacc = len(ta & tb) / len(ta | tb)
                if jacc >= jaccard_threshold:
                    consumed.add(b.memory_id)
                    cluster.append(b)

            if len(cluster) == 1:
                continue

            # Keep the longest-content member (richest fact); union tags.
            best = max(cluster, key=lambda m: len(m.content))
            cluster_tags: set[str] = set()
            for m in cluster:
                cluster_tags.update(m.tags)

            for m in cluster:
                if m.memory_id != best.memory_id:
                    self.delete(m.memory_id)
                    removed += 1

            merged = best.model_copy(update={"tags": sorted(cluster_tags)})
            self.upsert(merged)

        return removed
