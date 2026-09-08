"""
CivilityAI: Database Session & Persistence Layer.

Provides SQLAlchemy engine management, connection pooling, SQLite/PostgreSQL
compatibility, and FastAPI dependency injection for database sessions.
"""

from __future__ import annotations

import logging
import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api_service.entities.models import Base
from safety_ml.settings import ApplicationConfig

logger = logging.getLogger("CivilityAI.Persistence")

# Central configuration
_app_config = ApplicationConfig()
DATABASE_URL = _app_config.database_url

# Configure engine with dialect-specific options
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )

SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database() -> None:
    """
    Creates all database tables if they do not already exist.
    """
    logger.info(f"Initializing database schema at: {DATABASE_URL}")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized successfully.")


def get_database_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding an isolated transaction session per request.
    """
    session: Session = SessionFactory()
    try:
        yield session
    finally:
        session.close()
