"""API router aggregation."""

from fastapi import APIRouter

from src.api.documents import router as documents_router
from src.api.indexes import router as indexes_router
from src.api.search import router as search_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(indexes_router)
api_router.include_router(documents_router)
api_router.include_router(search_router)
