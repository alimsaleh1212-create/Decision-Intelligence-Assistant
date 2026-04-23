"""Convert raw thread message lists into ThreadChunk Pydantic objects.

Each chunk's text is a formatted dialogue string that reads naturally as a
conversation, giving the embedding model the best signal for retrieval.
The brand field is derived from the author_id of the first outbound message.
"""

import logging

from app.schemas.ingest import ThreadChunk, ThreadMessage

logger = logging.getLogger(__name__)

_CUSTOMER_LABEL = "[Customer]"
_BRAND_LABEL = "[Brand]"
_UNKNOWN_BRAND = "unknown"


def build_chunks(threads: list[list[ThreadMessage]]) -> list[ThreadChunk]:
    """Convert thread message lists into ThreadChunk objects ready for embedding.

    Args:
        threads: List of threads as returned by loader.reconstruct_batch().

    Returns:
        List of ThreadChunk objects, one per non-empty thread.
    """
    chunks: list[ThreadChunk] = []
    for messages in threads:
        chunk = _thread_to_chunk(messages)
        if chunk is not None:
            chunks.append(chunk)

    logger.info("Built %d chunks from %d threads", len(chunks), len(threads))
    return chunks


def _thread_to_chunk(messages: list[ThreadMessage]) -> ThreadChunk | None:
    """Format a single thread into a ThreadChunk.

    Brand is the author_id of the first outbound (brand) message in the thread.
    Falls back to "unknown" for customer-only threads with no brand reply yet.

    Args:
        messages: Ordered list of ThreadMessage for one thread.

    Returns:
        ThreadChunk, or None if messages is empty.
    """
    if not messages:
        return None

    brand = next(
        (msg.author_id for msg in messages if not msg.inbound),
        _UNKNOWN_BRAND,
    )

    lines = [
        f"{_CUSTOMER_LABEL if msg.inbound else _BRAND_LABEL}: {msg.text}"
        for msg in messages
    ]

    return ThreadChunk(
        thread_id=messages[0].tweet_id,
        brand=brand,
        text="\n".join(lines),
        message_count=len(messages),
        messages=messages,
    )
