"""Ticket retrieval service — embed query and search Qdrant.

Embedding is done via Ollama's embed endpoint using nomic-embed-text.
"""

import logging

import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.settings import get_settings
from app.schemas.query import RetrievedTicket

logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """Raised when embedding or vector search fails."""


async def retrieve(query: str) -> list[RetrievedTicket]:
    """Embed query and return top-k similar tickets from Qdrant.

    Args:
        query: The raw user query string.

    Returns:
        List of RetrievedTicket ordered by descending similarity score.

    Raises:
        RetrieverError: If embedding or Qdrant search fails.
    """
    settings = get_settings()
    vector = _embed(query, settings)
    return _search(vector, settings)


def _embed(text: str, settings: object) -> list[float]:
    """Embed text using Ollama's nomic-embed-text model.

    Args:
        text: Text to embed.
        settings: Application settings.

    Returns:
        Embedding vector as a list of floats.

    Raises:
        RetrieverError: If the Ollama embed call fails.
    """
    from app.core.settings import Settings

    assert isinstance(settings, Settings)
    try:
        client = ollama.Client(
            host=settings.ollama_base_url,
            timeout=settings.ollama_timeout_seconds,
        )
        response = client.embed(model=settings.ollama_embed_model, input=text)
        vector: list[float] = response.embeddings[0]
        logger.debug("Embedded query", extra={"dims": len(vector)})
        return vector
    except Exception as exc:
        logger.error("Embedding failed: %s", exc)
        raise RetrieverError(f"Embedding failed: {exc}") from exc


def _search(vector: list[float], settings: object) -> list[RetrievedTicket]:
    """Search Qdrant for the top-k nearest neighbours.

    Args:
        vector: Query embedding.
        settings: Application settings.

    Returns:
        List of RetrievedTicket sorted by score descending.

    Raises:
        RetrieverError: If Qdrant search fails.
    """
    from app.core.settings import Settings

    assert isinstance(settings, Settings)
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        hits = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            limit=settings.qdrant_top_k,
            with_payload=True,
        )
        tickets = [
            RetrievedTicket(
                text=hit.payload.get("text", "") if hit.payload else "",
                score=round(float(hit.score), 4),
            )
            for hit in hits
        ]
        logger.info("Qdrant search complete", extra={"hits": len(tickets)})
        return tickets
    except (UnexpectedResponse, Exception) as exc:
        logger.error("Qdrant search failed: %s", exc)
        raise RetrieverError(f"Qdrant search failed: {exc}") from exc
