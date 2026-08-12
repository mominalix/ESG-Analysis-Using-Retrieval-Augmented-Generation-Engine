"""Typed application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_TAXONOMY_PATH = ROOT_DIR / "config" / "esg_taxonomy.json"
PACKAGED_TAXONOMY_PATH = Path(__file__).with_name("esg_taxonomy.json")


class Settings(BaseSettings):
    """Runtime settings.

    Environment variables are the public configuration contract. Secrets use
    ``SecretStr`` so they cannot accidentally appear in logs or tracebacks.
    """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # Application
    api_title: str = "ESG Analysis RAG Platform"
    api_version: str = "3.0.0"
    api_description: str = "Evidence-grounded ESG document analysis using RAG"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False
    docs_enabled: bool = True
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    reload: bool = False

    # Security and HTTP
    secret_key: SecretStr | None = None
    admin_token: SecretStr | None = None
    enable_cors: bool = True
    allowed_origins_env: str = Field(
        default="http://localhost:8501",
        validation_alias=AliasChoices("ALLOWED_ORIGINS", "CORS_ALLOWED_ORIGINS"),
    )
    cors_allow_credentials: bool = False
    trusted_hosts_env: str = Field(default="", validation_alias="TRUSTED_HOSTS")

    # LLM providers
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    default_llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_model: str = Field(
        default="gpt-5.6-terra",
        validation_alias=AliasChoices("OPENAI_MODEL", "DEFAULT_MODEL"),
    )
    anthropic_model: str = "claude-sonnet-4-5-20250929"
    openai_use_responses_api: bool = True
    openai_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "low"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("MAX_OUTPUT_TOKENS", "MAX_TOKENS"),
    )

    # Vector stores. Memory is a real lexical store and keeps local setup
    # dependency-free; Chroma and Qdrant are opt-in production providers.
    vector_store_type: Literal["memory", "chroma", "qdrant"] = "memory"
    chroma_persist_directory: Path = ROOT_DIR / "data" / "chroma"
    chroma_collection_name: str = "esg_documents"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = "esg_documents"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=1536, ge=1)

    # Documents and taxonomy
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    max_chunks_per_document: int = Field(default=500, ge=1)
    supported_extensions_env: str = Field(
        default=".pdf,.docx,.txt,.md,.csv,.xlsx",
        validation_alias="SUPPORTED_EXTENSIONS",
    )
    upload_dir: Path = ROOT_DIR / "uploads"
    max_file_size: int = Field(default=10 * 1024 * 1024, ge=1)
    taxonomy_path: Path = (
        REPOSITORY_TAXONOMY_PATH if REPOSITORY_TAXONOMY_PATH.exists() else PACKAGED_TAXONOMY_PATH
    )

    # RAG
    top_k_results: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    max_context_length: int = Field(default=12_000, ge=500)
    enable_query_decomposition: bool = True
    enable_hybrid_search: bool = True
    enable_reranking: bool = False

    # Observability
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "esg-analysis"
    langsmith_tracing_v2: bool = False
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "json"
    enable_metrics: bool = True

    # UI/API client
    esg_api_url: str = "http://localhost:8000"
    api_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    api_read_timeout_seconds: float = Field(default=120.0, gt=0)
    api_retry_count: int = Field(default=2, ge=0, le=10)

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int, info):
        chunk_size = info.data.get("chunk_size", 1000)
        if value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        if self.environment == "production":
            if not self.admin_token or len(self.admin_token.get_secret_value()) < 24:
                raise ValueError("ADMIN_TOKEN must contain at least 24 characters in production")
            if "*" in self.allowed_origins:
                raise ValueError("Wildcard CORS origins are not allowed in production")
            if self.debug or self.reload:
                raise ValueError("DEBUG and RELOAD must be disabled in production")
        return self

    @property
    def allowed_origins(self) -> list[str]:
        return _csv_values(self.allowed_origins_env)

    @property
    def trusted_hosts(self) -> list[str]:
        return _csv_values(self.trusted_hosts_env)

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        values = []
        for extension in _csv_values(self.supported_extensions_env):
            normalized = extension.lower()
            values.append(normalized if normalized.startswith(".") else f".{normalized}")
        return tuple(values)

    # Backward-compatible read-only aliases used by older integrations.
    @property
    def default_model(self) -> str:
        return self.openai_model if self.default_llm_provider == "openai" else self.anthropic_model

    @property
    def max_tokens(self) -> int | None:
        return self.max_output_tokens

    @property
    def vector_db_provider(self) -> str:
        return self.vector_store_type

    @property
    def supported_file_types(self) -> list[str]:
        return list(self.supported_extensions)

    @property
    def max_document_size_mb(self) -> float:
        return self.max_file_size / (1024 * 1024)

    @property
    def embedding_model(self) -> str:
        return self.openai_embedding_model


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def reveal(secret: SecretStr | None) -> str | None:
    """Return a secret's value only at the provider boundary."""
    return secret.get_secret_value() if secret else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
