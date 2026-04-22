"""Load raw twcs.csv and reconstruct conversation threads via BFS.

Each thread starts from a customer's first-contact tweet (inbound=True,
no parent) and follows all replies depth-first up to MAX_DEPTH levels.
"""

import logging
from collections import deque
from pathlib import Path

import pandas as pd

from app.schemas.ingest import ThreadMessage

logger = logging.getLogger(__name__)

_MAX_DEPTH = 6


def load_threads(csv_path: str | Path, limit: int = 0) -> list[list[ThreadMessage]]:
    """Read twcs.csv and return conversation threads as ordered message lists.

    Args:
        csv_path: Absolute path to the raw twcs.csv file.
        limit: Cap on the number of threads returned. 0 means no cap.

    Returns:
        List of threads. Each thread is a list of ThreadMessage ordered from
        the root (customer first contact) to the deepest reply.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")

    required = {"tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"}
    df = pd.read_csv(csv_path, dtype={"tweet_id": int, "author_id": str, "text": str})

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    )
    df["text"] = df["text"].fillna("").str.strip()

    # Index tweets by ID for O(1) lookup
    tweet_by_id: dict[int, dict] = {
        row["tweet_id"]: row.to_dict() for _, row in df.iterrows()
    }

    # Build parent → children map from in_response_to_tweet_id
    children: dict[int, list[int]] = {tid: [] for tid in tweet_by_id}
    for row in tweet_by_id.values():
        parent = row["in_response_to_tweet_id"]
        if pd.notna(parent):
            parent_id = int(parent)
            if parent_id in children:
                children[parent_id].append(row["tweet_id"])

    # Root tweets: customer first contacts (no parent)
    roots = [
        tid
        for tid, row in tweet_by_id.items()
        if row["inbound"] and pd.isna(row["in_response_to_tweet_id"])
    ]
    logger.info("Found %d root (first-contact) tweets", len(roots))

    threads: list[list[ThreadMessage]] = []
    for root_id in roots:
        thread = _bfs_thread(root_id, tweet_by_id, children)
        if thread:
            threads.append(thread)
        if limit > 0 and len(threads) >= limit:
            break

    logger.info("Built %d conversation threads (limit=%d)", len(threads), limit)
    return threads


def _bfs_thread(
    root_id: int,
    tweet_by_id: dict[int, dict],
    children: dict[int, list[int]],
) -> list[ThreadMessage]:
    """Walk a thread via BFS from root_id, returning an ordered message list.

    Args:
        root_id: ID of the first-contact tweet.
        tweet_by_id: Mapping from tweet_id to row dict.
        children: Mapping from tweet_id to list of child tweet_ids.

    Returns:
        Ordered list of ThreadMessage from root to last reply (max _MAX_DEPTH deep).
    """
    messages: list[ThreadMessage] = []
    queue: deque[tuple[int, int]] = deque([(root_id, 0)])  # (tweet_id, depth)

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
                author_id=str(row["author_id"]),
                inbound=bool(row["inbound"]),
                text=str(row["text"]),
            )
        )

        for child_id in children.get(tid, []):
            if child_id not in visited:
                queue.append((child_id, depth + 1))

    return messages
