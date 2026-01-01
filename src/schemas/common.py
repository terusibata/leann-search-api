"""Common schemas for API responses."""

from enum import Enum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Error codes for API responses."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    INDEX_NOT_FOUND = "INDEX_NOT_FOUND"
    INDEX_ALREADY_EXISTS = "INDEX_ALREADY_EXISTS"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorDetail(BaseModel):
    """Error detail model."""

    code: ErrorCode
    message: str


T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""

    success: bool
    data: T | None = None
    error: ErrorDetail | None = None

    @classmethod
    def ok(cls, data: T) -> "APIResponse[T]":
        """Create a successful response."""
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(cls, code: ErrorCode, message: str) -> "APIResponse[Any]":
        """Create a failed response."""
        return cls(success=False, data=None, error=ErrorDetail(code=code, message=message))


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    leann_version: str
    embedding_model: str


class PaginationInfo(BaseModel):
    """Pagination information."""

    page: int
    per_page: int
    total: int
    total_pages: int
