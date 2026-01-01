"""Search-related schemas."""

from typing import Any

from pydantic import BaseModel, Field


class SearchOptions(BaseModel):
    """Search options."""

    search_complexity: int = Field(default=32, ge=16, le=256)
    include_content: bool = True
    include_metadata: bool = True
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchRequest(BaseModel):
    """Request for semantic search."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    metadata_filters: dict[str, Any] | None = None
    options: SearchOptions | None = None


class SearchResult(BaseModel):
    """Single search result."""

    document_id: str
    chunk_id: str
    content: str | None = None
    metadata: dict[str, Any] | None = None
    score: float
    position: int


class SearchData(BaseModel):
    """Search response data."""

    results: list[SearchResult]
    total_found: int
    query_time_ms: int


class SearchResponse(SearchData):
    """Response for semantic search."""

    pass


class GrepSearchRequest(BaseModel):
    """Request for keyword search."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=20, ge=1, le=100)
    metadata_filters: dict[str, Any] | None = None


class GrepSearchResult(BaseModel):
    """Single grep search result."""

    document_id: str
    chunk_id: str
    content: str
    metadata: dict[str, Any] | None = None
    match_positions: list[list[int]]


class GrepSearchData(BaseModel):
    """Grep search response data."""

    results: list[GrepSearchResult]
    total_found: int
    query_time_ms: int


class GrepSearchResponse(GrepSearchData):
    """Response for grep search."""

    pass


class HybridSearchRequest(BaseModel):
    """Request for hybrid search."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    metadata_filters: dict[str, Any] | None = None


class BatchSearchQuery(BaseModel):
    """Single query in batch search."""

    id: str
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class BatchSearchRequest(BaseModel):
    """Request for batch search."""

    queries: list[BatchSearchQuery] = Field(..., min_length=1, max_length=50)
    metadata_filters: dict[str, Any] | None = None


class BatchSearchResultItem(BaseModel):
    """Results for a single query in batch."""

    results: list[SearchResult]
    total_found: int


class BatchSearchData(BaseModel):
    """Batch search response data."""

    results: dict[str, BatchSearchResultItem]
    total_query_time_ms: int


class BatchSearchResponse(BatchSearchData):
    """Response for batch search."""

    pass
