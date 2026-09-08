"""
CivilityAI: Unit Tests for Database Persistence & Entity Models.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api_service.entities.models import (
    Base,
    ContentRecord,
    ReviewOutcome,
    SafetyAssessment,
)


@pytest.fixture
def memory_db_session():
    """In-memory SQLite database session fixture."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    session_maker = sessionmaker(bind=engine)
    session = session_maker()
    yield session
    session.close()


def test_content_record_and_safety_assessment_persistence(memory_db_session):
    """Verifies relational linkage between ContentRecord and SafetyAssessment."""
    # Create content record
    record = ContentRecord(message_body="Stop editing this page, you idiot.")
    memory_db_session.add(record)
    memory_db_session.commit()

    assert record.record_id is not None

    # Attach assessment
    assessment = SafetyAssessment(
        content_record_id=record.record_id,
        general_toxicity_score=0.91,
        severe_abuse_score=0.10,
        obscene_language_score=0.20,
        threatening_language_score=0.05,
        personal_insult_score=0.88,
        identity_attack_score=0.02,
        safety_risk_index=0.82,
        moderation_action="REVIEW",
        engine_version="CivilityAI-v1.0.0",
        processing_time_ms=25.4,
    )
    memory_db_session.add(assessment)
    memory_db_session.commit()

    # Query through relationship
    queried_record = memory_db_session.query(ContentRecord).filter_by(record_id=record.record_id).first()
    assert queried_record is not None
    assert queried_record.assessment is not None
    assert queried_record.assessment.moderation_action == "REVIEW"
    assert queried_record.assessment.safety_risk_index == 0.82


def test_review_outcome_moderation_audit(memory_db_session):
    """Verifies moderator audit action logging on content record."""
    record = ContentRecord(message_body="Severe violation message")
    memory_db_session.add(record)
    memory_db_session.commit()

    outcome = ReviewOutcome(
        content_record_id=record.record_id,
        review_action="REMOVE",
        reviewer_identifier="mod_user_7",
        notes="Violated Community Harassment Rule 3",
    )
    memory_db_session.add(outcome)
    memory_db_session.commit()

    queried_outcomes = memory_db_session.query(ReviewOutcome).filter_by(content_record_id=record.record_id).all()
    assert len(queried_outcomes) == 1
    assert queried_outcomes[0].review_action == "REMOVE"
    assert queried_outcomes[0].reviewer_identifier == "mod_user_7"
