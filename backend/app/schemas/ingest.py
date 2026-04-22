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
        text: Formatted conversation string used as the embedding input.
        message_count: Number of individual tweets in the thread.
        messages: Ordered list of tweets from first contact to last reply.
    """

    thread_id: int
    text: str
    message_count: int = Field(ge=1)
    messages: list[ThreadMessage]


class IngestRequest(BaseModel):
    """Request body for POST /api/ingest.

    Attributes:
        limit: Maximum number of threads to embed and store.
                Pass 0 to ingest all discovered threads.
        raw_csv_path: Optional override for the raw CSV path inside the container.
    """

    limit: int = Field(default=5_000, ge=0, description="Max threads to ingest; 0 = all.")
    raw_csv_path: str = Field(
        default="/app/data/raw/twcs.csv",
        description="Absolute path to twcs.csv inside the container.",
    )


class IngestResponse(BaseModel):
    """Immediate response returned when the ingest job is accepted.

    Attributes:
        message: Human-readable acceptance message.
        threads_requested: Number of threads the job will attempt to embed.
    """

    message: str
    threads_requested: int


class IngestStatusResponse(BaseModel):
    """Current state of the running (or last completed) ingest job.

    Attributes:
        status: One of "idle", "running", "done", "error".
        threads_total: Total threads scheduled for this run.
        threads_embedded: Threads successfully embedded so far.
        qdrant_count: Total points currently in the Qdrant collection.
        error: Error message if status is "error"; otherwise None.
    """

    status: str
    threads_total: int
    threads_embedded: int
    qdrant_count: int
    error: str | None = None
