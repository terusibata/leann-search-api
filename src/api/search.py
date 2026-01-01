"""Search API endpoints."""

from fastapi import APIRouter, HTTPException, status

from src.config import get_settings
from src.schemas.common import APIResponse, ErrorCode
from src.schemas.search import (
    BatchSearchRequest,
    BatchSearchResponse,
    GrepSearchRequest,
    GrepSearchResponse,
    HybridSearchRequest,
    SearchRequest,
    SearchResponse,
)
from src.services.index_service import get_index_service
from src.services.search_service import get_search_service

router = APIRouter(prefix="/indexes/{index_name}/search", tags=["search"])


def _check_index_exists(index_name: str) -> None:
    """Check if index exists, raise 404 if not."""
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


@router.post("", response_model=APIResponse[SearchResponse])
async def semantic_search(index_name: str, request: SearchRequest):
    """Perform semantic search."""
    _check_index_exists(index_name)

    settings = get_settings()
    options = request.options or {}

    search_complexity = getattr(options, "search_complexity", settings.search_complexity)
    include_content = getattr(options, "include_content", True)
    include_metadata = getattr(options, "include_metadata", True)
    min_score = getattr(options, "min_score", 0.0)

    top_k = min(request.top_k, settings.max_top_k)

    service = get_search_service()
    results, total_found, query_time_ms = service.search(
        index_name,
        request.query,
        top_k=top_k,
        metadata_filters=request.metadata_filters,
        search_complexity=search_complexity,
        include_content=include_content,
        include_metadata=include_metadata,
        min_score=min_score,
    )

    return APIResponse.ok(
        SearchResponse(
            results=results,
            total_found=total_found,
            query_time_ms=query_time_ms,
        )
    )


@router.post("/grep", response_model=APIResponse[GrepSearchResponse])
async def grep_search(index_name: str, request: GrepSearchRequest):
    """Perform keyword/grep search."""
    _check_index_exists(index_name)

    settings = get_settings()
    top_k = min(request.top_k, settings.max_top_k)

    service = get_search_service()
    results, total_found, query_time_ms = service.grep_search(
        index_name,
        request.query,
        top_k=top_k,
        metadata_filters=request.metadata_filters,
    )

    return APIResponse.ok(
        GrepSearchResponse(
            results=results,
            total_found=total_found,
            query_time_ms=query_time_ms,
        )
    )


@router.post("/hybrid", response_model=APIResponse[SearchResponse])
async def hybrid_search(index_name: str, request: HybridSearchRequest):
    """Perform hybrid search (semantic + keyword)."""
    _check_index_exists(index_name)

    settings = get_settings()
    top_k = min(request.top_k, settings.max_top_k)

    service = get_search_service()
    results, total_found, query_time_ms = service.hybrid_search(
        index_name,
        request.query,
        top_k=top_k,
        semantic_weight=request.semantic_weight,
        keyword_weight=request.keyword_weight,
        metadata_filters=request.metadata_filters,
    )

    return APIResponse.ok(
        SearchResponse(
            results=results,
            total_found=total_found,
            query_time_ms=query_time_ms,
        )
    )


@router.post("/batch", response_model=APIResponse[BatchSearchResponse])
async def batch_search(index_name: str, request: BatchSearchRequest):
    """Perform batch search for multiple queries."""
    _check_index_exists(index_name)

    queries = [q.model_dump() for q in request.queries]

    service = get_search_service()
    results, total_query_time_ms = service.batch_search(
        index_name,
        queries,
        metadata_filters=request.metadata_filters,
    )

    return APIResponse.ok(
        BatchSearchResponse(
            results=results,
            total_query_time_ms=total_query_time_ms,
        )
    )
