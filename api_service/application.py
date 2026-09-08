"""
CivilityAI: Production FastAPI Application.

Initializes database persistence, loads the ContentSafetyEngine once via lifespan,
configures structured logging, CORS, and registers modular routers.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api_service.endpoints import (
    analytics_router,
    batch_router,
    health_router,
    review_router,
    safety_router,
)
from api_service.persistence import init_database
from safety_ml.safety_inference import ContentSafetyEngine
from safety_ml.settings import ApplicationConfig

# Configure structured application logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ReqID: %(request_id)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CivilityAI.API")


@asynccontextmanager
async def application_lifespan(app: FastAPI):
    """
    Application lifespan manager: executes pre-warming operations on startup
    and clean teardown on shutdown.
    """
    logger.info("Initializing CivilityAI backend services...", extra={"request_id": "STARTUP"})

    # 1. Initialize DB schema (SQLite / PostgreSQL)
    init_database()

    # 2. Warm up and load ContentSafetyEngine singleton ONCE into memory
    engine = ContentSafetyEngine.get_singleton_instance()
    logger.info(
        f"Safety inference engine ready. Version: {engine.engine_version}, Device: {engine.device}",
        extra={"request_id": "STARTUP"},
    )

    yield

    logger.info("CivilityAI backend shutdown complete.", extra={"request_id": "SHUTDOWN"})


# Instantiate FastAPI application
api_application = FastAPI(
    title="CivilityAI: Content Safety & Moderation Platform",
    description=(
        "Production-grade Trust & Safety API for multi-label toxic language detection, "
        "risk calculation, and automated moderation governance."
    ),
    version="1.0.0",
    lifespan=application_lifespan,
)

# CORS Configuration for React Client
api_application.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits requests from Vite/React development and production servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api_application.middleware("http")
async def structured_logging_middleware(request: Request, call_next) -> Response:
    """
    Attaches unique request identifier and tracks end-to-end processing latency.
    Logs structured telemetry without persisting sensitive raw user text payloads.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    start_time = time.perf_counter()

    # Set logger context
    extra_context = {"request_id": request_id}

    response = await call_next(request)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Processing-Time-MS"] = str(elapsed_ms)

    # Structured audit logging
    if not request.url.path.endswith("/health"):
        logger.info(
            f"Method={request.method} Path={request.url.path} Status={response.status_code} Latency={elapsed_ms}ms",
            extra=extra_context,
        )

    return response


# Exception Handler
@api_application.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = request.headers.get("X-Request-ID", "UNKNOWN")
    logger.error(f"Unhandled Exception: {str(exc)}", exc_info=True, extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content={
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred during content evaluation.",
            "request_identifier": request_id,
        },
    )


# Mount Endpoint Routers
api_application.include_router(health_router)
api_application.include_router(safety_router)
api_application.include_router(review_router)
api_application.include_router(analytics_router)
api_application.include_router(batch_router)


@api_application.get("/", tags=["Root"])
def root_summary():
    return {
        "service": "CivilityAI",
        "description": "Intelligent Toxic Content Detection & Moderation Platform",
        "documentation": "/docs",
        "endpoints": {
            "health": "/system/health",
            "analyze": "/safety/analyze",
            "review_pending": "/review/pending",
            "review_action": "/review/{message_id}/action",
            "analytics": "/analytics/summary",
            "batch_analyze": "/batch/analyze",
        },
    }


if __name__ == "__main__":
    import uvicorn
    app_config = ApplicationConfig()
    uvicorn.run(
        "api_service.application:api_application",
        host=app_config.api_host,
        port=app_config.api_port,
        reload=False,
    )
