"""FastAPI application entry point."""

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.router import api_router
from src.config import get_settings
from src.schemas.common import APIResponse, ErrorCode, HealthResponse

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

settings = get_settings()

app = FastAPI(
    title="LEANN Search API",
    description="LEANN-based vector search API for RAG applications",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    errors = exc.errors()
    error_messages = []
    for error in errors:
        loc = " -> ".join(str(l) for l in error["loc"])
        error_messages.append(f"{loc}: {error['msg']}")

    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": "; ".join(error_messages),
            },
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": "An internal error occurred",
            },
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        leann_version=settings.leann_version,
        embedding_model=settings.embedding_model,
    )


# Include API router
app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info(
        "Starting LEANN Search API",
        version=settings.app_version,
        index_dir=settings.index_dir,
        embedding_model=settings.embedding_model,
    )

    # Ensure index directory exists
    settings.index_path.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down LEANN Search API")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
    )
