"""Chunker — passthrough for pre-formatted ThreadChunk objects.

Thread formatting (dialogue text, brand extraction) is done once in
notebooks/knowledge_preprocessing.ipynb. At ingest time chunks arrive
already formatted from loader.build_index(); this module exists as a
thin validation/logging layer so the router pipeline stays uniform.
"""

import logging

from app.schemas.ingest import ThreadChunk

logger = logging.getLogger(__name__)


def build_chunks(chunks: list[ThreadChunk]) -> list[ThreadChunk]:
    """Validate and return chunks, dropping any with empty text.

    Args:
        chunks: ThreadChunk objects as returned by loader.get_batch().

    Returns:
        Filtered list with empty-text chunks removed.
    """
    valid = [c for c in chunks if c.text.strip()]
    dropped = len(chunks) - len(valid)
    if dropped:
        logger.warning("Dropped %d chunks with empty text", dropped)
    logger.info("Chunks ready for embedding: %d", len(valid))
    return valid
