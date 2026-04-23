"""Pydantic schemas for the /api/query endpoint."""

from pydantic import BaseModel, Field


class RetrievedTicket(BaseModel):
    """A single ticket returned from the vector store."""

    text: str
    score: float = Field(ge=0.0, le=1.0)
    brand: str = ""


class QueryRequest(BaseModel):
    """Incoming query from the user."""

    query: str = Field(min_length=1, max_length=2000)
    brand: str | None = None


class QueryResponse(BaseModel):
    """Full response containing both answers and source tickets."""

    query: str
    rag_answer: str
    non_rag_answer: str
    retrieved_tickets: list[RetrievedTicket]


class BrandsResponse(BaseModel):
    """Distinct brand handles present in the vector store."""

    brands: list[str]
