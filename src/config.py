"""Configuration management."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "info"

    # LEANN settings
    index_dir: str = "./data/indexes"
    leann_backend: str = "hnsw"
    graph_degree: int = 32
    build_complexity: int = 64
    search_complexity: int = 32

    # Embedding settings
    embedding_model: str = "cl-nagoya/ruri-v3-310m"
    embedding_mode: str = "sentence-transformers"

    # Document settings
    default_chunk_size: int = 512
    default_chunk_overlap: int = 64
    max_upload_size_mb: int = 10

    # Search settings
    default_top_k: int = 10
    max_top_k: int = 100

    # App info
    app_version: str = "1.0.0"
    leann_version: str = "0.3.5"

    @property
    def index_path(self) -> Path:
        """Get index directory as Path."""
        return Path(self.index_dir)

    @property
    def max_upload_size_bytes(self) -> int:
        """Get max upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
