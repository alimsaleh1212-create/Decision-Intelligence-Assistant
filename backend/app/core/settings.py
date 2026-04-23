"""Application settings loaded once from environment variables.

All configuration lives here. Every other module imports from this file.
No os.getenv() calls anywhere else in the codebase.
"""

import logging
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Validated application configuration sourced from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Gemini — primary LLM
    google_api_key: str | None = None
    gemini_llm_model: str = "gemini-2.5-flash"

    # Ollama — fallback LLM + embeddings (embeddings always via Ollama)
    ollama_base_url: str = "http://ollama:11434"
    ollama_llm_model: str = "gemma4:31b-cloud"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout_seconds: int = 30

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "support_tickets"
    qdrant_top_k: int = 3

    # Backend
    log_level: str = "INFO"
    log_dir: str = "/app/logs"
    model_dir: str = "/app/models"

    # RAG retrieval quality gate — cosine similarity range [0, 1].
    # Chunks whose best match score falls below this value are considered
    # too dissimilar to be useful; the LLM call is skipped entirely.
    # Override with RAG_SCORE_THRESHOLD env var.
    rag_score_threshold: float = 0.6

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_url(cls, v: str) -> str:
        """Ensure Ollama URL is not empty."""
        if not v:
            raise ValueError("OLLAMA_BASE_URL must not be empty")
        return v

    @field_validator("qdrant_top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        """Ensure top-k is a positive integer."""
        if v < 1:
            raise ValueError("QDRANT_TOP_K must be >= 1")
        return v

    @property
    def gemini_configured(self) -> bool:
        """True when GOOGLE_API_KEY is set and non-empty."""
        return bool(self.google_api_key)


# Returned verbatim when no retrieved chunk meets the score threshold.
# Declared here (not in generator.py) so all modules share one source of truth.
RAG_NO_DATA_RESPONSE = (
    "We're sorry, but we couldn't find any relevant information in our knowledge base "
    "to answer your question. Please contact our support team directly for further assistance."
)


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings instance (loaded once per process)."""
    settings = Settings()
    logger.info(
        "Settings loaded",
        extra={
            "gemini_model": settings.gemini_llm_model,
            "gemini_configured": settings.gemini_configured,
            "ollama_model": settings.ollama_llm_model,
            "embed_model": settings.ollama_embed_model,
            "qdrant_host": settings.qdrant_host,
        },
    )
    return settings
