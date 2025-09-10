"""Application configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    environment: str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO", description="Log level")
    debug: bool = Field(default=False, description="Enable debug mode")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    api_key_header: str = Field(default="X-API-Key")
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:8501"])
    max_upload_size_mb: int = Field(default=50)

    # JWT
    jwt_secret_key: str = Field(default="change-me-in-production-use-256-bit-random-key")
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=60)
    jwt_refresh_token_expire_days: int = Field(default=30)

    # Qdrant Vector Store
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_grpc_port: int = Field(default=6334)
    qdrant_api_key: str | None = Field(default=None)
    qdrant_prefer_grpc: bool = Field(default=True)
    collection_name: str = Field(default="rag_documents")

    # Embeddings
    embedding_model: str = Field(default="sentence-transformers/all-mpnet-base-v2")
    embedding_batch_size: int = Field(default=32)
    embedding_device: str = Field(default="cpu")  # "cpu" | "cuda" | "mps"

    # Redis Cache
    redis_url: str = Field(default="redis://localhost:6379/0")
    cache_ttl_seconds: int = Field(default=3600)
    cache_max_connections: int = Field(default=20)

    # Database (for audit logs, user management)
    database_url: str = Field(default="sqlite+aiosqlite:///./rag.db")

    # Observability
    otlp_endpoint: str | None = Field(default=None, description="OpenTelemetry collector endpoint")
    prometheus_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090)

    # Rate limiting
    rate_limit_requests_per_minute: int = Field(default=60)
    rate_limit_burst: int = Field(default=10)

    # Chunking defaults
    default_chunk_size: int = Field(default=512)
    default_chunk_overlap: int = Field(default=100)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
