"""Local Qwen2.5-7B-Instruct backend for the trainable planner.

Loads the base model once on Apple Silicon (MPS) and supports hot-swapping
per-user LoRA adapters via PEFT for SDPO-fine-tuned planning.

Heavy ML dependencies (``torch``, ``transformers``, ``peft``) are imported
lazily so importing :mod:`nanoresearch.llm` on a machine without them works.
"""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from ..config import Device, get_settings
from ..logging import get_logger
from .base import ChatMessage, CompletionResult, LLMBackend

if TYPE_CHECKING:  # pragma: no cover - imports for type-checkers only
    from peft import PeftModel
    from transformers import AutoTokenizer, PreTrainedModel

_logger = get_logger(__name__)


class LocalQwenClient(LLMBackend):
    """Qwen2.5-7B-Instruct on Apple MPS with optional LoRA adapter."""

    name = "local-qwen"

    def __init__(
        self,
        *,
        model_path: str | None = None,
        device: Device | None = None,
        dtype: str = "float16",
    ) -> None:
        settings = get_settings()
        self.model_path = model_path or settings.qwen_model_path
        self.device = device or settings.local_model_device
        self.dtype = dtype
        self._model: PreTrainedModel | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._active_adapter: str | None = None
        self._lock = Lock()

    # ------------------------------------------------------------------ load

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch_dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[self.dtype]

            _logger.info(
                "loading_qwen",
                model_path=self.model_path,
                device=self.device,
                dtype=self.dtype,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=False
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            ).to(self.device.value)
            self._model.eval()

    # ------------------------------------------------------------- adapters

    def load_adapter(self, user_id: str, adapter_path: str | Path) -> None:
        """Attach (or refresh) a PEFT LoRA adapter for ``user_id``."""
        from peft import PeftModel

        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

        self._ensure_loaded()
        assert self._model is not None
        with self._lock:
            if isinstance(self._model, PeftModel):
                if user_id in self._model.peft_config:
                    self._model.delete_adapter(user_id)
                self._model.load_adapter(str(adapter_path), adapter_name=user_id)
            else:
                self._model = PeftModel.from_pretrained(
                    self._model, str(adapter_path), adapter_name=user_id
                )
            self._model.set_adapter(user_id)
            self._active_adapter = user_id
        _logger.info("adapter_loaded", user_id=user_id, path=str(adapter_path))

    def disable_adapter(self) -> None:
        """Run the next call against the base (non-LoRA) weights."""
        from peft import PeftModel

        if self._model is None or not isinstance(self._model, PeftModel):
            self._active_adapter = None
            return
        with self._lock:
            self._model.disable_adapter_layers()
            self._active_adapter = None

    def enable_adapter(self, user_id: str | None = None) -> None:
        from peft import PeftModel

        if self._model is None or not isinstance(self._model, PeftModel):
            return
        with self._lock:
            self._model.enable_adapter_layers()
            if user_id is not None:
                self._model.set_adapter(user_id)
                self._active_adapter = user_id

    # -------------------------------------------------------------- inference

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        response_format: dict[str, Any] | None = None,  # noqa: ARG002 - parity w/ base
        seed: int | None = None,
        extra: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> CompletionResult:
        import torch

        self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None

        prompt = self._tokenizer.apply_chat_template(
            [m.as_dict() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )

        if seed is not None:
            torch.manual_seed(seed)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device.value)
        prompt_tokens = int(inputs.input_ids.shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens or get_settings().planner_max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-5),
            "top_p": top_p,
            "pad_token_id": self._tokenizer.eos_token_id,
        }

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        new_tokens = output_ids[0, prompt_tokens:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
                    break

        completion_tokens = int(new_tokens.shape[0])
        _logger.debug(
            "qwen_complete",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 1),
            adapter=self._active_adapter,
        )
        return CompletionResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop",
            backend=self.name,
            model=self.model_path,
            latency_ms=latency_ms,
            raw={"adapter": self._active_adapter},
        )
