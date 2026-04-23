"""Batch embed ThreadChunk texts via the Ollama embeddings API.

Uses batch embedding (one API call per batch) to avoid the per-item overhead
of calling the embed endpoint 50K times sequentially.
"""

import logging

import ollama

from app.schemas.ingest import ThreadChunk

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64
# nomic-embed-text: 8192-token context. Tweet text is token-dense (URLs, @mentions,
# hashtags each expand to many subword tokens). Binary-search testing on real data
# shows ~7000 chars already hits the limit. 4000 chars is the safe ceiling.
# Full text is still stored in the Qdrant payload — only the embed input is truncated.
_MAX_EMBED_CHARS = 4000


def embed_chunks(
    chunks: list[ThreadChunk],
    ollama_client: ollama.Client,
    model: str,
) -> list[list[float]]:
    """Embed the text of each chunk using the Ollama embeddings API.

    Truncates each text to _MAX_EMBED_CHARS before embedding. If a batch still
    fails (e.g. a single chunk is pathologically dense), falls back to embedding
    one chunk at a time and skips any that still exceed the context limit.

    Args:
        chunks: ThreadChunk objects whose .text fields are to be embedded.
        ollama_client: Configured Ollama client instance.
        model: Name of the Ollama embedding model (e.g. "nomic-embed-text").

    Returns:
        List of embedding vectors, one per chunk, in the same order as chunks.
        Skipped chunks get a zero vector so indices stay aligned with chunks.

    Raises:
        RuntimeError: If the Ollama API returns no embeddings for a batch.
    """
    vectors: list[list[float]] = []
    total = len(chunks)

    for start in range(0, total, _BATCH_SIZE):
        batch = chunks[start : start + _BATCH_SIZE]
        texts = [c.text[:_MAX_EMBED_CHARS] for c in batch]

        try:
            resp = ollama_client.embed(model=model, input=texts)
            if not resp.embeddings:
                raise RuntimeError(
                    f"Ollama returned no embeddings for batch starting at index {start}"
                )
            vectors.extend(resp.embeddings)

        except Exception as batch_err:
            # Batch failed — fall back to one-by-one so a single bad chunk
            # does not block the rest of the batch.
            logger.warning(
                "Batch embed failed at index %d (%s) — retrying one-by-one",
                start,
                batch_err,
            )
            batch_vectors = _embed_one_by_one(batch, texts, ollama_client, model)
            vectors.extend(batch_vectors)

        logger.info(
            "Embedded batch %d–%d / %d",
            start,
            min(start + _BATCH_SIZE - 1, total - 1),
            total,
        )

    return vectors


def _embed_one_by_one(
    batch: list[ThreadChunk],
    texts: list[str],
    ollama_client: ollama.Client,
    model: str,
) -> list[list[float]]:
    """Embed each chunk individually; replace failures with a zero vector.

    Args:
        batch: The chunk slice that failed as a batch.
        texts: Pre-truncated texts corresponding to each chunk.
        ollama_client: Configured Ollama client instance.
        model: Embedding model name.

    Returns:
        List of vectors (zero vector for any chunk that still fails).
    """
    results: list[list[float]] = []
    for chunk, text in zip(batch, texts):
        try:
            resp = ollama_client.embed(model=model, input=[text])
            results.append(resp.embeddings[0])
        except Exception as e:
            logger.warning(
                "Skipping chunk thread_id=%d (char_len=%d): %s",
                chunk.thread_id,
                len(text),
                e,
            )
            results.append([])   # placeholder; filtered out before upsert
    return results
