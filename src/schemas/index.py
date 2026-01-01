"""Index-related schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class IndexSettings(BaseModel):
    """Index configuration settings."""

    backend: Literal["hnsw", "diskann"] = "hnsw"
    embedding_model: str = "cl-nagoya/ruri-v3-310m"
    graph_degree: int = Field(default=32, ge=8, le=128)
    build_complexity: int = Field(default=64, ge=32, le=512)
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0, le=512)


class IndexInfo(BaseModel):
    """Index information."""

    name: str
    created_at: datetime
    updated_at: datetime | None = None
    document_count: int = 0
    chunk_count: int = 0
    status: Literal["empty", "building", "ready", "error"] = "empty"
    settings: IndexSettings


class IndexStatistics(BaseModel):
    """Index statistics."""

    total_characters: int = 0
    avg_chunk_size: float = 0.0
    metadata_fields: list[str] = Field(default_factory=list)


class IndexDetailInfo(IndexInfo):
    """Detailed index information."""

    statistics: IndexStatistics | None = None


class IndexListData(BaseModel):
    """Index list data."""

    indexes: list[IndexInfo]
    total: int


class IndexListResponse(BaseModel):
    """Response for listing indexes."""

    indexes: list[IndexInfo]
    total: int


class CreateIndexRequest(BaseModel):
    """Request to create a new index."""

    name: str = Field(..., pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$", min_length=1, max_length=64)
    settings: IndexSettings | None = None


class CreateIndexResponse(BaseModel):
    """Response for creating an index."""

    name: str
    created_at: datetime
    status: str
    settings: IndexSettings


class IndexDetailResponse(IndexDetailInfo):
    """Response for getting index details."""

    pass


class IndexDeleteResponse(BaseModel):
    """Response for deleting an index."""

    message: str


class IndexRebuildRequest(BaseModel):
    """Request to rebuild an index."""

    settings: IndexSettings | None = None


class IndexRebuildResponse(BaseModel):
    """Response for rebuilding an index."""

    message: str
    chunk_count: int
    build_time_ms: int
