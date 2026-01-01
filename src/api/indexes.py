"""Index management API endpoints."""

from fastapi import APIRouter, HTTPException, status

from src.schemas.common import APIResponse, ErrorCode
from src.schemas.index import (
    CreateIndexRequest,
    CreateIndexResponse,
    IndexDeleteResponse,
    IndexDetailResponse,
    IndexListResponse,
    IndexRebuildRequest,
    IndexRebuildResponse,
    IndexSettings,
)
from src.services.index_service import get_index_service

router = APIRouter(prefix="/indexes", tags=["indexes"])


@router.get("", response_model=APIResponse[IndexListResponse])
async def list_indexes():
    """List all indexes."""
    service = get_index_service()
    indexes = service.list_indexes()

    return APIResponse.ok(
        IndexListResponse(indexes=indexes, total=len(indexes))
    )


@router.post("", response_model=APIResponse[CreateIndexResponse])
async def create_index(request: CreateIndexRequest):
    """Create a new index."""
    service = get_index_service()

    if service.index_exists(request.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "success": False,
                "data": None,
                "error": {
                    "code": ErrorCode.INDEX_ALREADY_EXISTS,
                    "message": f"Index '{request.name}' already exists",
                },
            },
        )

    index_info = service.create_index(request.name, request.settings)

    return APIResponse.ok(
        CreateIndexResponse(
            name=index_info.name,
            created_at=index_info.created_at,
            status=index_info.status,
            settings=index_info.settings,
        )
    )


@router.get("/{index_name}", response_model=APIResponse[IndexDetailResponse])
async def get_index(index_name: str):
    """Get index details."""
    service = get_index_service()
    index_info = service.get_index(index_name)

    if not index_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "success": False,
                "data": None,
                "error": {
                    "code": ErrorCode.INDEX_NOT_FOUND,
                    "message": f"Index '{index_name}' not found",
                },
            },
        )

    return APIResponse.ok(index_info)


@router.delete("/{index_name}", response_model=APIResponse[IndexDeleteResponse])
async def delete_index(index_name: str):
    """Delete an index."""
    service = get_index_service()

    if not service.index_exists(index_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "success": False,
                "data": None,
                "error": {
                    "code": ErrorCode.INDEX_NOT_FOUND,
                    "message": f"Index '{index_name}' not found",
                },
            },
        )

    service.delete_index(index_name)

    return APIResponse.ok(
        IndexDeleteResponse(message=f"Index '{index_name}' deleted successfully")
    )


@router.post("/{index_name}/rebuild", response_model=APIResponse[IndexRebuildResponse])
async def rebuild_index(index_name: str, request: IndexRebuildRequest | None = None):
    """Rebuild an index."""
    service = get_index_service()

    if not service.index_exists(index_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "success": False,
                "data": None,
                "error": {
                    "code": ErrorCode.INDEX_NOT_FOUND,
                    "message": f"Index '{index_name}' not found",
                },
            },
        )

    settings = request.settings if request else None
    chunk_count, build_time_ms = service.rebuild_index(index_name, settings)

    return APIResponse.ok(
        IndexRebuildResponse(
            message="Index rebuild completed",
            chunk_count=chunk_count,
            build_time_ms=build_time_ms,
        )
    )
