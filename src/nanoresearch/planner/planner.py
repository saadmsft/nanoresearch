"""High-level Planner interface.

Wraps the local Qwen2.5-7B model with per-user LoRA management. The
Orchestrator's ``Plan(...)`` calls (paper Eq. 1, 2, 5, 9) go through this
class. SDPO updates (Eq. 14-15) are dispatched via :meth:`update_from_feedback`.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

from ..config import Device, get_settings
from ..llm.base import ChatMessage
from ..logging import get_logger
from .adapters import AdapterManager
from .sdpo import SDPOConfig, SDPOExample, SDPOTrainer

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

_log = get_logger(__name__)


@dataclass
class PlannerOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    user_id: str | None


class Planner:
    """Manages the base Qwen model + a stack of per-user LoRA adapters."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        device: Device | None = None,
        dtype: str = "float16",
        adapter_manager: AdapterManager | None = None,
    ) -> None:
        s = get_settings()
        self.model_path = model_path or s.qwen_model_path
        self.device = (device or s.local_model_device).value
        self.dtype = dtype
        self.adapter_manager = adapter_manager or AdapterManager()

        self._base_model: PreTrainedModel | None = None
        self._peft_model: Any | None = None  # wraps base once first adapter attached
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._active_user: str | None = None
        self._lock = Lock()

    # ============================================================== load

    def _ensure_loaded(self) -> None:
        if self._base_model is not None:
            return
        with self._lock:
            if self._base_model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[self.dtype]

            _log.info(
                "loading_qwen",
                model_path=self.model_path,
                device=self.device,
                dtype=self.dtype,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=False
            )
            self._base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            ).to(self.device)

    @property
    def model(self) -> Any:
        """The currently-active model (PEFT-wrapped if any adapter loaded)."""
        return self._peft_model if self._peft_model is not None else self._base_model

    # ======================================================== adapters

    def ensure_user_adapter(self, user_id: str) -> None:
        """Attach the user's LoRA adapter, creating a fresh one if missing.

        Idempotent. After this call the active adapter is ``user_id``.
        """
        self._ensure_loaded()
        assert self._base_model is not None

        with self._lock:
            from peft import PeftModel

            # Fast path: already active.
            if self._active_user == user_id and self._peft_model is not None:
                return

            target = self._peft_model if self._peft_model is not None else self._base_model

            if self.adapter_manager.exists(user_id):
                self._peft_model = self.adapter_manager.attach_existing(target, user_id)
            else:
                self._peft_model = self.adapter_manager.attach_new(target, user_id)

            # PEFT may have wrapped or merely added an adapter.
            assert isinstance(self._peft_model, PeftModel)
            self._peft_model.set_adapter(user_id)
            self._active_user = user_id

    def disable_adapter(self) -> None:
        """Run subsequent calls against the base (non-LoRA) weights."""
        if self._peft_model is None:
            return
        with self._lock:
            self._peft_model.disable_adapter_layers()
            self._active_user = None

    # ======================================================== inference

    def plan(
        self,
        messages: list[ChatMessage],
        *,
        user_id: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        seed: int | None = None,
    ) -> PlannerOutput:
        """Generate one plan. If ``user_id`` is given, that user's adapter is used."""
        import time

        import torch

        self._ensure_loaded()
        assert self._tokenizer is not None

        if user_id is not None:
            self.ensure_user_adapter(user_id)

        m = self.model
        m.eval()

        prompt = self._tokenizer.apply_chat_template(
            [msg.as_dict() for msg in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        if seed is not None:
            torch.manual_seed(seed)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs.input_ids.shape[1])

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = m.generate(
                **inputs,
                max_new_tokens=max_new_tokens or get_settings().planner_max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0

        new_tokens = output_ids[0, prompt_tokens:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return PlannerOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(new_tokens.shape[0]),
            latency_ms=latency_ms,
            user_id=self._active_user,
        )

    # ======================================================== training

    def update_from_feedback(
        self,
        *,
        user_id: str,
        examples: list[SDPOExample],
        config: SDPOConfig | None = None,
    ) -> dict[str, float]:
        """Run an SDPO update for ``user_id`` and persist the adapter."""
        self.ensure_user_adapter(user_id)
        assert self._tokenizer is not None and self._peft_model is not None
        trainer = SDPOTrainer(
            self._peft_model,
            self._tokenizer,
            user_id=user_id,
            device=self.device,
            config=config,
            adapter_manager=self.adapter_manager,
        )
        summary = trainer.train(examples)
        trainer.save()
        return summary
