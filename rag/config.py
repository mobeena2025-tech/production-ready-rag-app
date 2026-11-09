from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str | None = None
    # "gemini" calls Google's API. "local" uses offline stand-ins (for demos and tests).
    provider: str = "gemini"
    chat_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"

    chroma_dir: str = "./data/chroma"
    collection: str = "documents"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4
    # Chunks less similar than this (cosine, 0-1) are ignored.
    # Leave unset to use the embedder's own default.
    min_score: float | None = None
    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
