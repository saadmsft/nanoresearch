"""Shared LLM types used by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass
class CompletionResult:
    """Result of a single chat completion."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    backend: str = ""
    model: str = ""
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


class LLMBackend(ABC):
    """Protocol every backend implements."""

    name: str = "abstract"

    @abstractmethod
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
        """Run one chat completion and return the assistant text + usage."""
