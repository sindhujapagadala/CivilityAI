"""
CivilityAI: Relational Database Models (SQLAlchemy).

Entities:
- ContentRecord: Ingested user content log
- SafetyAssessment: Detailed multi-category safety scores & moderation action
- ReviewOutcome: Human moderator audit trail and resolution
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


class ContentRecord(Base):
    """
    Stores original user-submitted content with submission timestamp.
    """
    __tablename__ = "content_records"

    record_id = Column(String(36), primary_key=True, default=generate_uuid)
    message_body = Column(Text, nullable=False)
    submitted_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    assessment = relationship("SafetyAssessment", back_populates="content_record", uselist=False, cascade="all, delete-orphan")
    review_outcomes = relationship("ReviewOutcome", back_populates="content_record", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ContentRecord(id={self.record_id}, length={len(self.message_body)})>"


class SafetyAssessment(Base):
    """
    Stores ML inference outputs, category risk probabilities, Safety Risk Index,
    and automated moderation action.
    """
    __tablename__ = "safety_assessments"

    assessment_id = Column(String(36), primary_key=True, default=generate_uuid)
    content_record_id = Column(String(36), ForeignKey("content_records.record_id", ondelete="CASCADE"), nullable=False, unique=True)

    general_toxicity_score = Column(Float, nullable=False, default=0.0)
    severe_abuse_score = Column(Float, nullable=False, default=0.0)
    obscene_language_score = Column(Float, nullable=False, default=0.0)
    threatening_language_score = Column(Float, nullable=False, default=0.0)
    personal_insult_score = Column(Float, nullable=False, default=0.0)
    identity_attack_score = Column(Float, nullable=False, default=0.0)

    safety_risk_index = Column(Float, nullable=False, index=True)
    moderation_action = Column(String(20), nullable=False, index=True)  # 'ALLOW', 'REVIEW', 'ESCALATE'
    engine_version = Column(String(50), nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    assessed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    content_record = relationship("ContentRecord", back_populates="assessment")

    def __repr__(self) -> str:
        return f"<SafetyAssessment(id={self.assessment_id}, action={self.moderation_action}, risk={self.safety_risk_index:.2f})>"


class ReviewOutcome(Base):
    """
    Audit record for human moderator reviews and actions.
    """
    __tablename__ = "review_outcomes"

    review_id = Column(String(36), primary_key=True, default=generate_uuid)
    content_record_id = Column(String(36), ForeignKey("content_records.record_id", ondelete="CASCADE"), nullable=False)

    review_action = Column(String(20), nullable=False)  # 'ALLOW', 'REMOVE', 'ESCALATE'
    reviewer_identifier = Column(String(100), nullable=False, default="system_moderator")
    notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    content_record = relationship("ContentRecord", back_populates="review_outcomes")

    def __repr__(self) -> str:
        return f"<ReviewOutcome(id={self.review_id}, action={self.review_action}, reviewer={self.reviewer_identifier})>"
