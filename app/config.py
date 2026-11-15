from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration from environment variables (or a local `.env` file)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://rag:rag@127.0.0.1:5432/rag"

    # Claude. Without ANTHROPIC_API_KEY the SDK falls back to its usual credential lookup.
    anthropic_api_key: SecretStr | None = None
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    claude_max_tokens: int = 16000

    # Optional semantic search. Without a key the app uses keyword search only.
    voyage_api_key: SecretStr | None = None
    voyage_model: str = "voyage-4"

    top_k: int = Field(default=6, ge=1, le=20)
    chunk_size: int = Field(default=1200, ge=200)
    chunk_overlap: int = Field(default=200, ge=0)

    # When set, every request needs a matching X-API-Key header.
    api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
