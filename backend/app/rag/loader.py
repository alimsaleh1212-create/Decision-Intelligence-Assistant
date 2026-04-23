"""Load preprocessed thread chunks from data/knowledge/thread_chunks.csv.

The heavy ETL (BFS thread reconstruction, adjacency map, brand extraction) is done
once in notebooks/knowledge_preprocessing.ipynb and saved to thread_chunks.csv.
This module only reads that file and returns ThreadChunk objects — no graph traversal.

Two-phase design matches the batch-aware router:
  build_index() — load full CSV once, keep in memory.
  get_batch()   — slice a range of rows by cursor position.
"""

import logging
from pathlib import Path

import pandas as pd

from app.schemas.ingest import ThreadChunk, ThreadMessage

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"thread_id", "brand", "message_count", "text"}


def build_index(csv_path: str | Path) -> list[ThreadChunk]:
    """Load thread_chunks.csv and return all chunks as ThreadChunk objects.

    This is called once per process. The result is cached in router state
    and sliced by get_batch() for each successive ingest batch.

    Args:
        csv_path: Absolute path to thread_chunks.csv.

    Returns:
        List of ThreadChunk objects, one per row.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Knowledge CSV not found: {csv_path}\n"
            "Run notebooks/knowledge_preprocessing.ipynb first."
        )

    logger.info("Loading knowledge CSV from %s…", csv_path)
    df = pd.read_csv(csv_path, dtype={"thread_id": int, "brand": str, "text": str})

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"thread_chunks.csv is missing columns: {missing}")

    df["text"]          = df["text"].fillna("").str.strip()
    df["brand"]         = df["brand"].fillna("unknown").str.strip()
    df["message_count"] = df["message_count"].fillna(1).astype(int)

    chunks = [
        ThreadChunk(
            thread_id=int(row["thread_id"]),
            brand=str(row["brand"]),
            text=str(row["text"]),
            message_count=int(row["message_count"]),
            messages=[],  # raw messages not stored in CSV — not needed for embedding
        )
        for _, row in df.iterrows()
    ]

    logger.info("Loaded %d thread chunks", len(chunks))
    return chunks


def get_batch(all_chunks: list[ThreadChunk], cursor: int, batch_size: int) -> list[ThreadChunk]:
    """Return the next slice of chunks starting at cursor.

    Args:
        all_chunks: Full list as returned by build_index().
        cursor: Index of the first chunk in this batch.
        batch_size: Maximum number of chunks to return.

    Returns:
        A slice of ThreadChunk objects (may be shorter than batch_size at the end).
    """
    return all_chunks[cursor : cursor + batch_size]
