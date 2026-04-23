"""Pydantic schemas for the ingest API.

Defines the data shapes for conversation thread chunks, request, and response models.
"""

from pydantic import BaseModel, Field


class ThreadMessage(BaseModel):
    """A single tweet within a conversation thread.

    Attributes:
        tweet_id: Original tweet ID from the dataset.
        author_id: The handle of the tweet's author.
        inbound: True when the author is a customer; False when the author is the brand.
        text: Raw tweet text.
    """

    tweet_id: int
    author_id: str
    inbound: bool
    text: str


class ThreadChunk(BaseModel):
    """A complete conversation thread formatted as a single RAG chunk.

    Each chunk captures one full customer–brand exchange starting from
    the customer's first-contact tweet and following all replies.

    Attributes:
        thread_id: Deterministic identifier derived from the root tweet ID.
        brand: Handle of the brand the customer is communicating with (e.g. "sprintcare").
               Derived from the author_id of the first outbound message in the thread.
               Falls back to "unknown" for threads with no brand reply.
        text: Formatted conversation string used as the embedding input.
        message_count: Number of individual tweets in the thread.
        messages: Ordered list of tweets from first contact to last reply.
    """

    thread_id: int
    brand: str
    text: str
    message_count: int = Field(ge=1)
    messages: list[ThreadMessage]


class IngestRequest(BaseModel):
    """Request body for POST /api/ingest.

    Attributes:
        batch_size: Number of thread roots to embed and store in this call.
        raw_csv_path: Absolute path to twcs.csv inside the container (or locally).
        reset: If True, discard the cached index and cursor, restart from the beginning.
    """

    batch_size: int = Field(default=500, ge=1, description="Threads to process per call.")
    raw_csv_path: str = Field(
        default="/app/data/raw/twcs.csv",
        description="Absolute path to twcs.csv.",
    )
    reset: bool = Field(default=False, description="Restart ingest from the beginning.")


class IngestResponse(BaseModel):
    """Immediate response returned when a batch is accepted.

    Attributes:
        message: Human-readable acceptance message.
        batch_number: Which batch this is (1-indexed).
        roots_remaining: How many roots are still unprocessed after this batch completes.
    """

    message: str
    batch_number: int
    roots_remaining: int


class IngestStatusResponse(BaseModel):
    """Current state of the ingest pipeline.

    Attributes:
        status: One of "idle", "indexing", "running", "ready", "done", "error".
        total_roots: Total thread roots discovered (set after first index build).
        cursor: Number of roots processed so far across all batches.
        batches_completed: How many POST /api/ingest calls have finished successfully.
        threads_embedded: Cumulative threads embedded across all batches.
        qdrant_count: Total points currently in the Qdrant collection.
        error: Error message if status is "error"; otherwise None.
    """

    status: str
    total_roots: int
    cursor: int
    batches_completed: int
    threads_embedded: int
    qdrant_count: int
    error: str | None = None
