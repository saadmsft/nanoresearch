"""Routes each agent role to the appropriate LLM backend.

Per the implementation plan: the **planner** runs on the local Qwen2.5-7B
client (because SDPO needs token-level logits + gradient updates), and every
other agent runs on GPT-5.1 via Azure AI Foundry.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

from ..logging import RunManifest, get_logger
from .azure_foundry import AzureFoundryClient
from .base import ChatMessage, CompletionResult, LLMBackend

_logger = get_logger(__name__)


class AgentRole(StrEnum):
    """Stable identifiers for every agent in the pipeline."""

    PLANNER = "planner"  # ⟵ trainable; routes to LocalQwenClient
    IDEATION = "ideation"
    PLANNING = "planning"
    CODING = "coding"
    DEBUG = "debug"
    ANALYSIS = "analysis"
    WRITING = "writing"
    REVIEW = "review"
    REVISION = "revision"
    JUDGE_COMPLIANCE = "judge_compliance"
    JUDGE_NOVELTY = "judge_novelty"
    JUDGE_WRITING = "judge_writing"
    SIMULATED_USER = "simulated_user"


# Every role that does NOT need logit/gradient access goes to Azure.
_AZURE_ROLES: frozenset[AgentRole] = frozenset(r for r in AgentRole if r is not AgentRole.PLANNER)


class LLMRouter:
    """Routes ``(role, messages)`` to the right backend and records the call."""

    def __init__(
        self,
        *,
        azure: LLMBackend | None = None,
        planner: LLMBackend | None = None,
        manifest: RunManifest | None = None,
    ) -> None:
        # Azure is constructed lazily-eager (cheap) — the local Qwen backend is
        # only built when explicitly required so importers without torch work.
        self._azure: LLMBackend = azure or AzureFoundryClient()
        self._planner: LLMBackend | None = planner
        self._manifest = manifest

    def _planner_backend(self) -> LLMBackend:
        if self._planner is None:
            # Lazy import: only pulls torch/transformers when planner is used.
            from .local_qwen import LocalQwenClient

            self._planner = LocalQwenClient()
        return self._planner

    def backend_for(self, role: AgentRole) -> LLMBackend:
        if role is AgentRole.PLANNER:
            return self._planner_backend()
        if role in _AZURE_ROLES:
            return self._azure
        raise ValueError(f"No backend mapped for role: {role}")

    def complete(
        self,
        role: AgentRole,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CompletionResult:
        backend = self.backend_for(role)
        start = time.perf_counter()
        result = backend.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            seed=seed,
            extra=extra,
        )
        # backend already records latency; the router records the role.
        if self._manifest is not None:
            preview = messages[-1].content if messages else ""
            self._manifest.record_llm_call(
                role=role.value,
                backend=result.backend,
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                latency_ms=result.latency_ms,
                prompt_preview=preview,
                completion_preview=result.text,
            )
        else:
            # Still emit a structured log line so dev can see the call.
            _logger.info(
                "llm_call",
                role=role.value,
                backend=result.backend,
                model=result.model,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                latency_ms=round(result.latency_ms, 1),
            )

        _ = time.perf_counter() - start
        return result
