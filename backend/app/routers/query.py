"""Query endpoint — RAG and non-RAG answer generation.

Accepts a user query, retrieves similar tickets from Qdrant,
and returns both a RAG answer (context-augmented) and a non-RAG answer
(LLM only) produced in parallel.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.core.settings import get_settings
from app.schemas.query import BrandsResponse, QueryRequest, QueryResponse
from app.services import generator, retriever
from app.services.query_logger import log_query
from app.services.retriever import RetrieverError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a user query with RAG and non-RAG responses.

    Args:
        request: Validated query payload (query text + optional brand filter).

    Returns:
        QueryResponse containing both answers and retrieved source tickets.

    Raises:
        HTTPException 500: If retrieval or generation fails unexpectedly.
    """
    logger.info(
        "Query received",
        extra={"query_length": len(request.query), "brand_filter": request.brand},
    )

    try:
        tickets = await retriever.retrieve(request.query, brand=request.brand)
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


@router.get("/brands", response_model=BrandsResponse)
async def brands() -> BrandsResponse:
    """Return distinct brand handles present in the Qdrant collection.

    Returns:
        BrandsResponse with a sorted list of brand strings.

    Raises:
        HTTPException 500: If the Qdrant facet query fails.
    """
    try:
        brand_list = retriever.get_distinct_brands(get_settings())
    except RetrieverError as exc:
        logger.error("Brands endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail="Could not fetch brands") from exc

    return BrandsResponse(brands=brand_list)
