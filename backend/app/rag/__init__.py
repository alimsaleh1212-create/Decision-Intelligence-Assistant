"""RAG sub-package: loader, chunker, embedder, store.

Each module is pure Python and HTTP-agnostic — no FastAPI imports here.
The ingest router orchestrates these modules but they remain independently testable.
"""
