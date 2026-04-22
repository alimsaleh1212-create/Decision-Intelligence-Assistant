"""Batch embed ThreadChunk texts via the Ollama embeddings API.

Uses batch embedding (one API call per batch) to avoid the per-item overhead
of calling the embed endpoint 50K times sequentially.
"""

import logging

import ollama

from app.schemas.ingest import ThreadChunk

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64


def embed_chunks(
    chunks: list[ThreadChunk],
    ollama_client: ollama.Client,
    model: str,
) -> list[list[float]]:
    """Embed the text of each chunk using the Ollama embeddings API.

    Args:
        chunks: ThreadChunk objects whose .text fields are to be embedded.
        ollama_client: Configured Ollama client instance.
        model: Name of the Ollama embedding model (e.g. "nomic-embed-text").

    Returns:
        List of embedding vectors, one per chunk, in the same order as chunks.

    Raises:
        RuntimeError: If the Ollama API returns no embeddings for a batch.
    """
    vectors: list[list[float]] = []
    total = len(chunks)

    for start in range(0, total, _BATCH_SIZE):
        batch = chunks[start : start + _BATCH_SIZE]
        texts = [c.text for c in batch]

        resp = ollama_client.embed(model=model, input=texts)
        if not resp.embeddings:
            raise RuntimeError(
                f"Ollama returned no embeddings for batch starting at index {start}"
            )

        vectors.extend(resp.embeddings)
        logger.info(
            "Embedded batch %d–%d / %d",
            start,
            min(start + _BATCH_SIZE - 1, total - 1),
            total,
        )

    return vectors
