"""Qdrant collection management and chunk upsert operations.

Payload stored per point: text, thread_id, brand, message_count.
No priority_label — classification is handled at query time by the ML model.
"""

import logging

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.schemas.ingest import ThreadChunk

logger = logging.getLogger(__name__)


def ensure_collection(client: QdrantClient, collection: str, dims: int) -> None:
    """Create the Qdrant collection if it does not already exist.

    Args:
        client: Qdrant client instance.
        collection: Target collection name.
        dims: Embedding vector dimensionality.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' (dims=%d)", collection, dims)
    else:
        logger.info("Collection '%s' exists — upserting into existing.", collection)


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[ThreadChunk],
    vectors: list[list[float]],
    batch_size: int = 256,
) -> int:
    """Upsert embedded chunks into the Qdrant collection in batches.

    Point ID is the thread_id (root tweet ID), a stable int64.
    Payload: text, thread_id, brand, message_count.

    Args:
        client: Qdrant client instance.
        collection: Target collection name.
        chunks: ThreadChunk objects to store.
        vectors: Corresponding embedding vectors (same order as chunks).
        batch_size: Number of points per upsert call.

    Returns:
        Total number of points now in the collection after upsert.

    Raises:
        ValueError: If chunks and vectors have different lengths.
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match vector count ({len(vectors)})"
        )

    # Drop any chunks whose embedding failed (empty vector placeholder from embedder)
    pairs = [(c, v) for c, v in zip(chunks, vectors) if v]
    skipped = len(chunks) - len(pairs)
    if skipped:
        logger.warning("Skipping %d chunks with empty embedding vectors", skipped)
    if not pairs:
        logger.warning("No valid vectors to upsert in this batch")
        return 0

    chunks, vectors = zip(*pairs)  # type: ignore[assignment]
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch_chunks = chunks[start : start + batch_size]
        batch_vectors = vectors[start : start + batch_size]

        points = [
            PointStruct(
                id=chunk.thread_id,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "thread_id": chunk.thread_id,
                    "brand": chunk.brand,
                    "message_count": chunk.message_count,
                },
            )
            for chunk, vector in zip(batch_chunks, batch_vectors)
        ]

        client.upsert(collection_name=collection, points=points)
        logger.info("Upserted points %d–%d / %d", start, start + len(points) - 1, total)

    count = client.count(collection_name=collection).count
    logger.info("Upsert complete. Collection '%s' now has %d points.", collection, count)
    return count


def get_collection_count(client: QdrantClient, collection: str) -> int:
    """Return the current point count for a collection, or 0 if it does not exist.

    Args:
        client: Qdrant client instance.
        collection: Collection name to query.

    Returns:
        Number of points in the collection, or 0 if the collection is absent.
    """
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        return 0
    return client.count(collection_name=collection).count
