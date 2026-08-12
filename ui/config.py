"""Frontend-only configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]


class UISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    esg_api_url: str = "http://localhost:8000"
    admin_token: SecretStr | None = None
    api_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    api_read_timeout_seconds: float = Field(default=120.0, gt=0)
    api_retry_count: int = Field(default=2, ge=0, le=10)

    @field_validator("esg_api_url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("ESG_API_URL must be an http(s) URL")
        return value


@lru_cache(maxsize=1)
def get_ui_settings() -> UISettings:
    return UISettings()


ui_settings = get_ui_settings()
