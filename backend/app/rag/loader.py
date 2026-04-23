"""Load raw twcs.csv and reconstruct conversation threads via BFS.

Two-phase design:
  1. build_index() — load the full CSV once; build adjacency map + root list.
     Keep the result in module/router state; never call this more than once per process.
  2. reconstruct_batch() — given a slice of root IDs, BFS-walk each thread.
     Call this repeatedly for successive batches without re-reading the CSV.

Thread roots are customer first-contact tweets (inbound=True, no parent).
Threads are walked depth-first up to MAX_DEPTH levels.
"""

import logging
from collections import deque
from pathlib import Path

import pandas as pd

from app.schemas.ingest import ThreadMessage

logger = logging.getLogger(__name__)

_MAX_DEPTH = 6
_USECOLS = ["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"]


def build_index(
    csv_path: str | Path,
) -> tuple[dict[int, dict], dict[int, list[int]], list[int]]:
    """Read twcs.csv and build the full adjacency index.

    This is the expensive step (~30s for 2.8M rows). Call it once and cache the
    result. The returned structures are reused by reconstruct_batch() for every
    subsequent batch without touching the disk again.

    Args:
        csv_path: Absolute path to the raw twcs.csv file.

    Returns:
        A 3-tuple:
          - tweet_by_id: {tweet_id → row dict} for O(1) lookup.
          - children: {tweet_id → [child tweet_ids]} adjacency map.
          - roots: Ordered list of root tweet_ids (customer first-contacts).

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    logger.info("Building index from %s — this may take ~30s for 2.8M rows…", csv_path)

    df = pd.read_csv(
        csv_path,
        usecols=_USECOLS,
        dtype={"tweet_id": int, "author_id": str, "text": str},
    )

    missing = set(_USECOLS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    )
    df["text"] = df["text"].fillna("").str.strip()
    df["inbound"] = df["inbound"].astype(bool)

    tweet_by_id: dict[int, dict] = {
        int(row["tweet_id"]): {
            "tweet_id": int(row["tweet_id"]),
            "author_id": str(row["author_id"]),
            "inbound": bool(row["inbound"]),
            "text": str(row["text"]),
            "parent": row["in_response_to_tweet_id"],
        }
        for _, row in df.iterrows()
    }

    children: dict[int, list[int]] = {tid: [] for tid in tweet_by_id}
    for row in tweet_by_id.values():
        parent = row["parent"]
        if pd.notna(parent):
            pid = int(parent)
            if pid in children:
                children[pid].append(row["tweet_id"])

    roots: list[int] = [
        tid
        for tid, row in tweet_by_id.items()
        if row["inbound"] and pd.isna(row["parent"])
    ]

    logger.info(
        "Index built: %d tweets, %d roots",
        len(tweet_by_id),
        len(roots),
    )
    return tweet_by_id, children, roots


def reconstruct_batch(
    root_ids: list[int],
    tweet_by_id: dict[int, dict],
    children: dict[int, list[int]],
) -> list[list[ThreadMessage]]:
    """Reconstruct conversation threads for a specific slice of root IDs.

    Designed to be called repeatedly with successive slices from the full roots list.
    Each call is O(threads_in_slice × avg_thread_depth) — no disk I/O.

    Args:
        root_ids: Slice of root tweet_ids to process in this batch.
        tweet_by_id: Full adjacency index as returned by build_index().
        children: Parent→children map as returned by build_index().

    Returns:
        List of threads. Each thread is a list of ThreadMessage ordered from
        root (customer first contact) to deepest reply.
    """
    threads: list[list[ThreadMessage]] = []
    for root_id in root_ids:
        thread = _bfs_thread(root_id, tweet_by_id, children)
        if thread:
            threads.append(thread)
    return threads


def _bfs_thread(
    root_id: int,
    tweet_by_id: dict[int, dict],
    children: dict[int, list[int]],
) -> list[ThreadMessage]:
    """BFS-walk a single thread from root_id, returning ordered messages.

    Args:
        root_id: ID of the first-contact tweet.
        tweet_by_id: Full tweet lookup dict.
        children: Parent→children adjacency map.

    Returns:
        Ordered list of ThreadMessage, root first (max _MAX_DEPTH deep).
    """
    messages: list[ThreadMessage] = []
    queue: deque[tuple[int, int]] = deque([(root_id, 0)])
    visited: set[int] = set()

    while queue:
        tid, depth = queue.popleft()
        if tid in visited or depth > _MAX_DEPTH:
            continue
        visited.add(tid)

        row = tweet_by_id.get(tid)
        if row is None:
            continue

        messages.append(
            ThreadMessage(
                tweet_id=tid,
                author_id=row["author_id"],
                inbound=row["inbound"],
                text=row["text"],
            )
        )

        for child_id in children.get(tid, []):
            if child_id not in visited:
                queue.append((child_id, depth + 1))

    return messages
