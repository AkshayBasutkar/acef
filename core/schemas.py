"""
ACEF Data Model — the entire data contract for the system.

One enum (QueryType), two request/response models, one intermediate chunk
model, and a health-check model. No activation maps, no execution plans.
"""

import time
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """Three-way classification of incoming queries."""
    SIMPLE = "simple"
    CODE = "code"
    MEMORY = "memory"


class QueryRequest(BaseModel):
    """Incoming query from the user."""
    query: str = Field(min_length=1, max_length=4096)
    session_id: UUID = Field(default_factory=uuid4)


class RetrievedChunk(BaseModel):
    """A single piece of context retrieved by GraphRAG or MemRAG."""
    text: str
    source: str  # filename, function name, or doc chunk id — shown to the user


class QueryResponse(BaseModel):
    """Final response returned to the user."""
    query_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    response: str
    query_type: QueryType
    from_cache: bool
    sources: list[str] = Field(default_factory=list)
    latency_ms: float


class HealthResponse(BaseModel):
    """Health-check response for the /health endpoint."""
    status: str
    neo4j_connected: bool
    chroma_ready: bool
