"""LLM backends and agent-role routing.

Two backends:
- :class:`AzureFoundryClient` — GPT-5.1 on Azure AI Foundry via AAD bearer token.
- :class:`LocalQwenClient` — Qwen2.5-7B-Instruct on Apple MPS with hot-swappable
  per-user LoRA adapters (planner role only).

A single :class:`LLMRouter` maps each agent role to the appropriate backend.
"""

from .azure_foundry import AzureFoundryClient
from .base import (
    ChatMessage,
    CompletionResult,
    LLMBackend,
    Role,
)
from .router import AgentRole, LLMRouter

__all__ = [
    "AgentRole",
    "AzureFoundryClient",
    "ChatMessage",
    "CompletionResult",
    "LLMBackend",
    "LLMRouter",
    "Role",
]
