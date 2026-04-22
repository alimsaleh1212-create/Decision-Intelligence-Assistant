"""Ingest processed tickets into Qdrant.

Reads data/processed/tickets.csv, embeds each ticket text via Ollama,
and upserts all points into the Qdrant collection.

Run once after the stack is up:
    docker compose exec backend python scripts/ingest.py
    docker compose exec backend python scripts/ingest.py --limit 50000

The --limit flag caps how many tickets are embedded (default: 50 000).
Embedding 775K rows one-by-one would take many hours; 50K gives strong
retrieval coverage in a manageable time.
"""

import argparse
import logging
import sys
from pathlib import Path

import ollama
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

# Allow running from the backend/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.settings import get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PROCESSED_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "tickets.csv"
_BATCH_SIZE = 50
_DEFAULT_LIMIT = 50_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest tickets into Qdrant.")
    parser.add_argument(
        "--limit",
        type=int,
        default=_DEFAULT_LIMIT,
        help=f"Max tickets to embed (default: {_DEFAULT_LIMIT:,}). Pass 0 for all rows.",
    )
    return parser.parse_args()


def main() -> None:
    """Embed and upsert processed tickets into Qdrant."""
    args = _parse_args()
    settings = get_settings()

    if not _PROCESSED_PATH.exists():
        logger.error("Processed data not found: %s — run the notebook first.", _PROCESSED_PATH)
        sys.exit(1)

    df = pd.read_csv(_PROCESSED_PATH)
    if "text" not in df.columns:
        logger.error("CSV must have a 'text' column; found: %s", list(df.columns))
        sys.exit(1)

    texts = df["text"].dropna().tolist()
    logger.info("Loaded %d tickets from %s", len(texts), _PROCESSED_PATH)

    if args.limit > 0 and len(texts) > args.limit:
        # Deterministic sample so re-runs are consistent
        import random
        random.seed(42)
        texts = random.sample(texts, args.limit)
        logger.info("Sampled %d tickets (--limit %d)", len(texts), args.limit)

    qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    ollama_client = ollama.Client(host=settings.ollama_base_url)

    # Probe embedding dims with one sample
    sample_resp = ollama_client.embed(model=settings.ollama_embed_model, input=texts[0])
    dims = len(sample_resp.embeddings[0])
    logger.info("Embedding model: %s, dims=%d", settings.ollama_embed_model, dims)

    _ensure_collection(qdrant, settings.qdrant_collection, dims)

    points: list[PointStruct] = []
    for idx, text in enumerate(texts):
        resp = ollama_client.embed(model=settings.ollama_embed_model, input=text)
        vector = resp.embeddings[0]
        points.append(PointStruct(id=idx, vector=vector, payload={"text": text}))

        if len(points) >= _BATCH_SIZE:
            qdrant.upsert(collection_name=settings.qdrant_collection, points=points)
            logger.info("Upserted batch up to idx=%d", idx)
            points.clear()

    if points:
        qdrant.upsert(collection_name=settings.qdrant_collection, points=points)

    count = qdrant.count(collection_name=settings.qdrant_collection).count
    logger.info("Ingest complete. Collection '%s' has %d points.", settings.qdrant_collection, count)


def _ensure_collection(client: QdrantClient, name: str, dims: int) -> None:
    """Create the Qdrant collection if it does not already exist.

    Args:
        client: Qdrant client instance.
        name: Collection name.
        dims: Embedding dimensionality.
    """
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' (dims=%d)", name, dims)
    else:
        logger.info("Collection '%s' already exists — upserting into existing.", name)


if __name__ == "__main__":
    main()
