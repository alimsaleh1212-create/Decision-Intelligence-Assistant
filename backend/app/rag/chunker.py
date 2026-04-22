"""Convert raw thread message lists into ThreadChunk Pydantic objects.

Each chunk's text is a formatted dialogue string that reads naturally as a
conversation, which gives the embedding model the best signal for retrieval.
"""

import logging

from app.schemas.ingest import ThreadChunk, ThreadMessage

logger = logging.getLogger(__name__)

_CUSTOMER_LABEL = "[Customer]"
_BRAND_LABEL = "[Brand]"


def build_chunks(threads: list[list[ThreadMessage]]) -> list[ThreadChunk]:
    """Convert thread message lists into ThreadChunk objects ready for embedding.

    Args:
        threads: List of threads as returned by loader.load_threads().

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

    Args:
        messages: Ordered list of ThreadMessage for one thread.

    Returns:
        ThreadChunk, or None if messages is empty.
    """
    if not messages:
        return None

    root = messages[0]
    lines: list[str] = []
    for msg in messages:
        label = _CUSTOMER_LABEL if msg.inbound else _BRAND_LABEL
        lines.append(f"{label}: {msg.text}")

    text = "\n".join(lines)

    return ThreadChunk(
        thread_id=root.tweet_id,
        text=text,
        message_count=len(messages),
        messages=messages,
    )
