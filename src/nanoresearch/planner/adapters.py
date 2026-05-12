"""Per-user LoRA adapter lifecycle.

Each user has a directory under ``data/users/<user_id>/lora/`` that holds a
PEFT ``LoraConfig`` + safetensors weights. The :class:`AdapterManager` knows
how to **create** (cold start), **load** (attach to base model), **save**
(persist after SDPO step), and **list** adapters.

All heavy imports (torch, peft) are lazy so callers without the local extras
installed can still import this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import get_settings
from ..logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from peft import LoraConfig
    from transformers import PreTrainedModel

_log = get_logger(__name__)


# Default LoRA hyperparameters — tuned for Qwen2.5-7B on M1 Max 32 GB
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class AdapterManager:
    """Manages per-user LoRA adapters on disk."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else get_settings().lora_adapters_dir

    # ------------------------------------------------------------- paths
    def user_lora_dir(self, user_id: str) -> Path:
        """Top-level LoRA dir for ``user_id``.

        PEFT nests one level deeper by adapter name, so the actual config
        and weights live at :meth:`adapter_path` (``<user_lora_dir>/<user_id>/``).
        """
        d = self.root / user_id / "lora"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def adapter_path(self, user_id: str) -> Path:
        return self.user_lora_dir(user_id) / user_id

    def exists(self, user_id: str) -> bool:
        return (self.adapter_path(user_id) / "adapter_config.json").exists()

    def list_users(self) -> list[str]:
        if not self.root.exists():
            return []
        out: list[str] = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / "lora" / d.name / "adapter_config.json").exists():
                out.append(d.name)
        return out

    # ------------------------------------------------------------- config
    def make_lora_config(
        self,
        *,
        rank: int = DEFAULT_LORA_RANK,
        alpha: int = DEFAULT_LORA_ALPHA,
        dropout: float = DEFAULT_LORA_DROPOUT,
        target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
    ) -> LoraConfig:
        from peft import LoraConfig, TaskType

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none",
            inference_mode=False,
        )

    # ------------------------------------------------------------- attach
    def attach_new(
        self,
        base_model: PreTrainedModel,
        user_id: str,
        *,
        lora_config: LoraConfig | None = None,
    ) -> Any:
        """Wrap ``base_model`` with a fresh LoRA adapter for ``user_id``.

        Returns the resulting :class:`peft.PeftModel`. Subsequent users can
        attach further adapters with :meth:`attach_existing` (or
        :meth:`load_existing`) without reloading the base.
        """
        from peft import PeftModel, get_peft_model

        cfg = lora_config or self.make_lora_config()

        if isinstance(base_model, PeftModel):
            base_model.add_adapter(user_id, cfg)
            base_model.set_adapter(user_id)
            peft_model = base_model
        else:
            peft_model = get_peft_model(base_model, cfg, adapter_name=user_id)

        _log.info("adapter_created", user_id=user_id, rank=cfg.r, alpha=cfg.lora_alpha)
        return peft_model

    def attach_existing(
        self,
        base_model: PreTrainedModel,
        user_id: str,
    ) -> Any:
        from peft import PeftModel

        path = self.adapter_path(user_id)
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(f"No saved adapter for user {user_id} at {path}")
        if isinstance(base_model, PeftModel):
            if user_id in base_model.peft_config:
                base_model.delete_adapter(user_id)
            base_model.load_adapter(str(path), adapter_name=user_id)
            base_model.set_adapter(user_id)
            return base_model
        return PeftModel.from_pretrained(base_model, str(path), adapter_name=user_id)

    def save(self, peft_model: Any, user_id: str) -> Path:
        """Persist ``peft_model``'s adapter for ``user_id`` to disk.

        PEFT's ``save_pretrained`` nests by adapter name, so we point it at
        :meth:`user_lora_dir` and the actual files land in
        :meth:`adapter_path`.
        """
        save_root = self.user_lora_dir(user_id)
        peft_model.save_pretrained(str(save_root), selected_adapters=[user_id])
        path = self.adapter_path(user_id)
        _log.info("adapter_saved", user_id=user_id, path=str(path))
        return path
