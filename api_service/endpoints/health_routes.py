"""
CivilityAI: System Health & Diagnostic Endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from api_service.contracts.schemas import SystemHealthResponse
from api_service.persistence import get_database_session
from safety_ml.safety_inference import ContentSafetyEngine

health_router = APIRouter(prefix="/system", tags=["System Diagnostics"])


@health_router.get("/health", response_model=SystemHealthResponse)
def check_system_health(
    session: Session = Depends(get_database_session),
) -> SystemHealthResponse:
    """
    Returns engine version, device information, and database connection status.
    """
    engine = ContentSafetyEngine.get_singleton_instance()

    # Verify database connectivity
    database_connected = False
    try:
        session.execute(text("SELECT 1"))
        database_connected = True
    except Exception:
        database_connected = False

    return SystemHealthResponse(
        status="healthy" if database_connected and engine.is_loaded else "degraded",
        engine_version=engine.engine_version,
        model_backend=engine.app_config.model_backend,
        device=str(engine.device),
        database_connected=database_connected,
        timestamp=datetime.now(timezone.utc),
    )
