"""Ticket retrieval service — embed query and search Qdrant.

Embedding is done via Ollama's embed endpoint using nomic-embed-text.
"""

import logging

import ollama
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import FieldCondition, Filter, MatchValue  # type: ignore[import]

from app.core.settings import get_settings
from app.schemas.query import RetrievedTicket

logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """Raised when embedding or vector search fails."""


async def retrieve(query: str, brand: str | None = None) -> list[RetrievedTicket]:
    """Embed query and return top-k similar tickets from Qdrant.

    Args:
        query: The raw user query string.
        brand: Optional brand handle to restrict results to (exact match on payload).

    Returns:
        List of RetrievedTicket ordered by descending similarity score.

    Raises:
        RetrieverError: If embedding or Qdrant search fails.
    """
    settings = get_settings()
    vector = _embed(query, settings)
    return _search(vector, settings, brand=brand)


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


def _search(
    vector: list[float],
    settings: object,
    brand: str | None = None,
) -> list[RetrievedTicket]:
    """Search Qdrant for the top-k nearest neighbours, with optional brand filter.

    Args:
        vector: Query embedding.
        settings: Application settings.
        brand: If set, restrict results to points whose payload brand matches exactly.

    Returns:
        List of RetrievedTicket sorted by score descending.

    Raises:
        RetrieverError: If Qdrant search fails.
    """
    from app.core.settings import Settings

    assert isinstance(settings, Settings)

    query_filter: Filter | None = None
    if brand:
        query_filter = Filter(
            must=[FieldCondition(key="brand", match=MatchValue(value=brand))]
        )

    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        hits = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=vector,
            query_filter=query_filter,
            limit=settings.qdrant_top_k,
            with_payload=True,
        )
        tickets = [
            RetrievedTicket(
                text=hit.payload.get("text", "") if hit.payload else "",
                score=round(float(hit.score), 4),
                brand=hit.payload.get("brand", "") if hit.payload else "",
            )
            for hit in hits
        ]
        logger.info(
            "Qdrant search complete",
            extra={"hits": len(tickets), "brand_filter": brand},
        )
        return tickets
    except (UnexpectedResponse, Exception) as exc:
        logger.error("Qdrant search failed: %s", exc)
        raise RetrieverError(f"Qdrant search failed: {exc}") from exc


def get_distinct_brands(settings: object) -> list[str]:
    """Return sorted list of distinct brand values in the Qdrant collection.

    Scrolls all points and collects unique brand values from the payload.
    Efficient enough for the current dataset size; use a payload index +
    facet API if the collection grows beyond ~100K points.

    Args:
        settings: Application settings.

    Returns:
        Sorted list of brand strings, empty list if collection is empty.

    Raises:
        RetrieverError: If the Qdrant call fails.
    """
    from app.core.settings import Settings

    assert isinstance(settings, Settings)
    try:
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        brands: set[str] = set()
        offset = None
        while True:
            result, next_offset = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=256,
                offset=offset,
                with_payload=["brand"],
                with_vectors=False,
            )
            for point in result:
                b = point.payload.get("brand", "") if point.payload else ""
                if b:
                    brands.add(b)
            if next_offset is None:
                break
            offset = next_offset
        brand_list = sorted(brands)
        logger.info("Brand scroll complete", extra={"count": len(brand_list)})
        return brand_list
    except Exception as exc:
        logger.error("Brand scroll failed: %s", exc)
        raise RetrieverError(f"Brand scroll failed: {exc}") from exc
