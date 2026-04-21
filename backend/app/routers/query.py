"""Query endpoint — RAG and non-RAG answer generation.

Accepts a user query, retrieves similar tickets from Qdrant,
and returns both a RAG answer (context-augmented) and a non-RAG answer
(LLM only) produced in parallel.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.query import QueryRequest, QueryResponse
from app.services import generator, retriever
from app.services.query_logger import log_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a user query with RAG and non-RAG responses.

    Args:
        request: Validated query payload.

    Returns:
        QueryResponse containing both answers and retrieved source tickets.

    Raises:
        HTTPException 500: If retrieval or generation fails unexpectedly.
    """
    logger.info("Query received", extra={"query_length": len(request.query)})

    try:
        tickets = await retriever.retrieve(request.query)
        rag_answer, non_rag_answer = await generator.generate_both(
            request.query, tickets
        )
    except Exception as exc:
        logger.error("Query pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail="Query processing failed") from exc

    response = QueryResponse(
        query=request.query,
        rag_answer=rag_answer,
        non_rag_answer=non_rag_answer,
        retrieved_tickets=tickets,
    )

    await log_query(request.query, tickets, rag_answer, non_rag_answer)
    return response
