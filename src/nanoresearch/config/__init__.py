"""Application-wide settings loaded from environment / .env file."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Device(StrEnum):
    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


class Settings(BaseSettings):
    """Top-level configuration.

    Values are read from environment variables (or a `.env` file in the
    working directory). All Azure access is via Azure AD — no API keys.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Azure AI Foundry (GPT-5.1) ---------------------------------------
    azure_openai_endpoint: HttpUrl = Field(
        ...,
        description="Foundry / Azure OpenAI endpoint URL.",
    )
    azure_openai_deployment: str = Field(
        ...,
        description="Deployment name for the GPT-5.1 model.",
    )
    azure_openai_api_version: str = Field(
        "2024-12-01-preview",
        description="Azure OpenAI REST API version.",
    )
    azure_openai_token_scope: str = Field(
        "https://cognitiveservices.azure.com/.default",
        description="AAD scope for bearer-token auth.",
    )

    # --- Local planner (Qwen2.5-7B-Instruct) ------------------------------
    qwen_model_path: str = Field(
        "Qwen/Qwen2.5-7B-Instruct",
        description="HF repo id or absolute path to a local snapshot.",
    )
    lora_adapters_dir: Path = Field(
        Path("./data/users"),
        description="Per-user LoRA adapter root (one subdir per user_id).",
    )
    local_model_device: Device = Field(
        Device.MPS,
        description="Torch device for the local model.",
    )
    planner_max_new_tokens: int = Field(1024, ge=1)

    # --- Storage / runs ---------------------------------------------------
    runs_dir: Path = Field(Path("./runs"))
    data_dir: Path = Field(Path("./data"))
    log_level: str = Field("INFO")

    @field_validator("azure_openai_deployment")
    @classmethod
    def _no_blank_deployment(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("AZURE_OPENAI_DEPLOYMENT must be set")
        return v.strip()

    @property
    def azure_endpoint_str(self) -> str:
        """Azure OpenAI client expects a str, not pydantic HttpUrl."""
        return str(self.azure_openai_endpoint).rstrip("/")


_settings: Settings | None = None


def get_settings() -> Settings:
    """Cached settings accessor (loads .env once)."""
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings


def reset_settings_cache() -> None:
    """Test hook: force re-read of the environment."""
    global _settings
    _settings = None
