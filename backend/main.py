# ai_ethics_toolkit/backend/main.py
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import logging
import sys
import uvicorn
import uuid

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings

# ------------------------------------------------------------------------------
#  Add project root to PYTHONPATH so "backend" imports resolve both in dev & gunicorn
# ------------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# ------------------------------------------------------------------------------
#  Application Settings (read once at start-up)
# ------------------------------------------------------------------------------

class Settings(BaseSettings):
    api_name: str = "AI Ethics Toolkit API"
    api_version: str = "1.0.0"

    # Networking
    host: str = "0.0.0.0"
    port: int = 12000  # Use port 12000 for deployment
    reload: bool = False

    # CORS
    allowed_origins: list[AnyHttpUrl] | list[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "https://work-1-fyddksdsujljwsjp.prod-runtime.all-hands.dev",
        "https://work-2-fyddksdsujljwsjp.prod-runtime.all-hands.dev",
        "*"
    ]

    # Toggle optional integrations
    enable_metrics: bool = False
    enable_tracing: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# ------------------------------------------------------------------------------
#  Logging (ISO 8601, service tag, structured fields)
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s.%(msecs)03dZ | %(levelname)s | "
        "%(name)s | %(request_id)s | %(message)s"
    ),
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ai_toolkit")

# Add a filter to provide a default request_id
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = '-'
        return True

logger.addFilter(RequestIdFilter())

# ------------------------------------------------------------------------------
#  Import routers & middleware AFTER path setup to avoid circulars
# ------------------------------------------------------------------------------
from backend.api.routes import (
    bias,
    privacy,
    explainability,
    hallucination,
    reports,
)
from backend.api.middleware.logging import RequestLoggingMiddleware
from backend.services.in_memory_service import InMemoryService

# ------------------------------------------------------------------------------
#  Lifespan handlers
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise shared resources on start-up and
    perform clean-up on shut-down.
    """
    logger.info("🚀 Booting %s v%s …", settings.api_name, settings.api_version)

    # ---------- start-up ----------
    # Initialize in-memory storage service
    app.state.storage = InMemoryService()
    logger.info("📦 In-memory storage initialized")

    logger.info("✅ All services initialised")
    yield

    # ---------- shut-down ----------
    logger.info("🔄 Shutting down …")
    # Clean up in-memory storage
    app.state.storage.clear()
    logger.info("🏁 Shutdown complete")


# ------------------------------------------------------------------------------
#  FastAPI application instance
# ------------------------------------------------------------------------------
app = FastAPI(
    title=settings.api_name,
    description="Comprehensive API for ethical AI development & auditing",
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ------------------------------------------------------------------------------
#  Middleware
# ------------------------------------------------------------------------------
app.add_middleware(GZipMiddleware, minimum_size=1_000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(url) for url in settings.allowed_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# ------------------------------------------------------------------------------
#  Routers
# ------------------------------------------------------------------------------
app.include_router(bias.router, prefix="/api/v1/bias", tags=["Bias Detection"])
app.include_router(privacy.router, prefix="/api/v1/privacy", tags=["Privacy Analysis"])
app.include_router(
    explainability.router, prefix="/api/v1/explainability", tags=["Explainability"]
)
app.include_router(
    hallucination.router, prefix="/api/v1/hallucination", tags=["Hallucination Detection"]
)
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])

# ------------------------------------------------------------------------------
#  Public endpoints
# ------------------------------------------------------------------------------


@app.get("/", tags=["Meta"])
async def root() -> dict[str, str]:
    return {
        "message": settings.api_name,
        "version": settings.api_version,
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "docs": "/docs",
    }


@app.get("/health", tags=["Meta"])
async def health_check() -> dict:
    """
    Aggregate health check.
    Signals liveness to load-balancers.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "storage": "online",
            "api": "online",
        },
    }


# ------------------------------------------------------------------------------
#  Session management
# ------------------------------------------------------------------------------
@app.get("/api/v1/session", tags=["Session"])
async def create_session() -> dict:
    """Create a new session for the client"""
    session_id = str(uuid.uuid4())
    app.state.storage.create_session(session_id)
    return {
        "session_id": session_id,
        "created_at": datetime.utcnow().isoformat(),
        "status": "active"
    }


# ------------------------------------------------------------------------------
#  Error handlers
# ------------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def fastapi_http_exception(
    request: Request, exc: HTTPException  # noqa: D401
):  # → JSON
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):  # → JSON
    logger.exception("Unhandled exception: %s", exc, extra={"request_id": request.headers.get("X-Request-ID", "-")})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ------------------------------------------------------------------------------
#  dev entry-point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info",
    )
