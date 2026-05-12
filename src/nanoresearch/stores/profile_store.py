"""Per-user :class:`UserProfile` persistence.

Path layout::

    data/users/<user_id>/profile.json
    data/users/<user_id>/skills/*.json    # populated by SkillBank
    data/users/<user_id>/memories/*.json  # populated by MemoryStore
    data/users/<user_id>/lora/            # populated by SDPO trainer
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..schemas import UserProfile


class ProfileStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def user_dir(self, user_id: str) -> Path:
        d = self.root / user_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def skills_dir(self, user_id: str) -> Path:
        d = self.user_dir(user_id) / "skills"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def memories_dir(self, user_id: str) -> Path:
        d = self.user_dir(user_id) / "memories"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def lora_dir(self, user_id: str) -> Path:
        d = self.user_dir(user_id) / "lora"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def profile_path(self, user_id: str) -> Path:
        return self.user_dir(user_id) / "profile.json"

    def save(self, profile: UserProfile) -> UserProfile:
        profile = profile.model_copy(update={"updated_at": datetime.now(UTC)})
        self.profile_path(profile.user_id).write_text(profile.model_dump_json(indent=2))
        return profile

    def load(self, user_id: str) -> UserProfile | None:
        path = self.profile_path(user_id)
        if not path.exists():
            return None
        return UserProfile.model_validate_json(path.read_text())

    def exists(self, user_id: str) -> bool:
        return self.profile_path(user_id).exists()

    def list_users(self) -> list[str]:
        return sorted(d.name for d in self.root.iterdir() if d.is_dir())
