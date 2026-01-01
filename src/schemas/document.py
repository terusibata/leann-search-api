"""Document-related schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.schemas.common import PaginationInfo


class DocumentInput(BaseModel):
    """Input for adding a document."""

    id: str | None = None
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] | None = None


class DocumentAddOptions(BaseModel):
    """Options for adding documents."""

    chunk_size: int | None = None
    chunk_overlap: int | None = None
    update_if_exists: bool = False


class AddDocumentsRequest(BaseModel):
    """Request to add documents."""

    documents: list[DocumentInput] = Field(..., min_length=1)
    options: DocumentAddOptions | None = None


class DocumentAddResult(BaseModel):
    """Result for a single document addition."""

    id: str
    chunk_count: int
    status: Literal["added", "updated", "failed"]
    error: str | None = None


class AddDocumentsData(BaseModel):
    """Data for add documents response."""

    added: int
    updated: int
    failed: int
    documents: list[DocumentAddResult]


class AddDocumentsResponse(AddDocumentsData):
    """Response for adding documents."""

    pass


class ChunkInfo(BaseModel):
    """Chunk information."""

    chunk_id: str
    content: str
    position: int


class DocumentInfo(BaseModel):
    """Basic document information."""

    id: str
    content_preview: str
    metadata: dict[str, Any] | None = None
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime | None = None


class DocumentListData(BaseModel):
    """Data for document list response."""

    documents: list[DocumentInfo]
    pagination: PaginationInfo


class DocumentListResponse(DocumentListData):
    """Response for listing documents."""

    pass


class DocumentDetail(BaseModel):
    """Detailed document information."""

    id: str
    content: str
    metadata: dict[str, Any] | None = None
    chunks: list[ChunkInfo] = Field(default_factory=list)
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime | None = None


class DocumentDetailResponse(DocumentDetail):
    """Response for getting document details."""

    pass


class UpdateDocumentOptions(BaseModel):
    """Options for updating a document."""

    merge_metadata: bool = True


class UpdateDocumentRequest(BaseModel):
    """Request to update a document."""

    content: str | None = None
    metadata: dict[str, Any] | None = None
    options: UpdateDocumentOptions | None = None


class UpdateDocumentResponse(BaseModel):
    """Response for updating a document."""

    id: str
    chunk_count: int
    status: str


class UpdateMetadataRequest(BaseModel):
    """Request to update metadata only."""

    metadata: dict[str, Any]
    merge: bool = True


class UpdateMetadataResponse(BaseModel):
    """Response for updating metadata."""

    id: str
    metadata: dict[str, Any]


class DeleteDocumentResponse(BaseModel):
    """Response for deleting a document."""

    message: str


class BulkDeleteRequest(BaseModel):
    """Request for bulk delete."""

    document_ids: list[str] | None = None
    metadata_filter: dict[str, Any] | None = None


class BulkDeleteResponse(BaseModel):
    """Response for bulk delete."""

    deleted: int
    message: str


class FileUploadResponse(BaseModel):
    """Response for file upload."""

    id: str
    filename: str
    file_type: str
    page_count: int | None = None
    chunk_count: int
    status: str
