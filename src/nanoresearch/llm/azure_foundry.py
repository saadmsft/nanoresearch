"""Azure AI Foundry / Azure OpenAI client using AAD bearer-token auth.

Key auth is disabled on the target resource — we authenticate via
``DefaultAzureCredential`` which transparently handles ``az login`` locally,
managed identity in cloud, env-creds in CI.

The caller (or operator) must have the **Cognitive Services OpenAI User**
role assignment on the resource.
"""

from __future__ import annotations

import time
from functools import cached_property
from typing import Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import APIError, AzureOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import get_settings
from ..logging import get_logger
from .base import ChatMessage, CompletionResult, LLMBackend

_logger = get_logger(__name__)


class AzureFoundryClient(LLMBackend):
    """GPT-5.1 deployment on Azure AI Foundry, AAD-authenticated."""

    name = "azure-foundry"

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        deployment: str | None = None,
        api_version: str | None = None,
        token_scope: str | None = None,
        credential: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.endpoint = endpoint or settings.azure_endpoint_str
        self.deployment = deployment or settings.azure_openai_deployment
        self.api_version = api_version or settings.azure_openai_api_version
        self.token_scope = token_scope or settings.azure_openai_token_scope
        self._credential = credential or DefaultAzureCredential()

    @cached_property
    def _client(self) -> AzureOpenAI:
        token_provider = get_bearer_token_provider(self._credential, self.token_scope)
        return AzureOpenAI(
            azure_endpoint=self.endpoint,
            azure_ad_token_provider=token_provider,
            api_version=self.api_version,
        )

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.5, min=1, max=30),
        reraise=True,
    )
    def complete(
        self,
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
        params: dict[str, Any] = {
            "model": self.deployment,
            "messages": [m.as_dict() for m in messages],
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            # GPT-5.1 / Azure 2024-12 preview uses max_completion_tokens; fallback
            # via `extra` allows callers to override for older deployments.
            params["max_completion_tokens"] = max_tokens
        if stop:
            params["stop"] = stop
        if response_format:
            params["response_format"] = response_format
        if seed is not None:
            params["seed"] = seed
        if extra:
            params.update(extra)

        start = time.perf_counter()
        resp = self._client.chat.completions.create(**params)
        latency_ms = (time.perf_counter() - start) * 1000.0

        choice = resp.choices[0]
        usage = resp.usage
        result = CompletionResult(
            text=choice.message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            finish_reason=choice.finish_reason,
            backend=self.name,
            model=self.deployment,
            latency_ms=latency_ms,
            raw={"id": resp.id},
        )
        _logger.debug(
            "azure_complete",
            deployment=self.deployment,
            latency_ms=round(latency_ms, 1),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
        return result
