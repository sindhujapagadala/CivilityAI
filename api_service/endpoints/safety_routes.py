"""
CivilityAI: Real-Time Content Safety Analysis Endpoint.
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api_service.contracts.schemas import (
    MessageAnalysisRequest,
    MessageAnalysisResponse,
    SafetyCategoryScores,
)
from api_service.entities.models import ContentRecord, SafetyAssessment
from api_service.persistence import get_database_session
from safety_ml.safety_inference import ContentSafetyEngine

logger = logging.getLogger("CivilityAI.SafetyRoutes")
safety_router = APIRouter(prefix="/safety", tags=["Safety Analysis"])


@safety_router.post(
    "/analyze",
    response_model=MessageAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
def analyze_content_safety(
    request: MessageAnalysisRequest,
    session: Session = Depends(get_database_session),
) -> MessageAnalysisResponse:
    """
    Analyzes submitted user content across all 6 toxicity categories, computes
    the Safety Risk Index, assigns automated moderation actions, and persists
    the assessment record.
    """
    engine = ContentSafetyEngine.get_singleton_instance()

    try:
        inference_result = engine.analyze_message(request.message_body)
    except Exception as exc:
        logger.error(f"Inference failure during content assessment: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference engine encountered an internal processing error.",
        )

    # Persist record and assessment in database
    content_record = ContentRecord(
        message_body=request.message_body,
    )
    session.add(content_record)
    session.flush()  # Generates record_id

    category_scores_data = inference_result["category_scores"]

    safety_assessment = SafetyAssessment(
        content_record_id=content_record.record_id,
        general_toxicity_score=category_scores_data["general_toxicity"],
        severe_abuse_score=category_scores_data["severe_abuse"],
        obscene_language_score=category_scores_data["obscene_language"],
        threatening_language_score=category_scores_data["threatening_language"],
        personal_insult_score=category_scores_data["personal_insult"],
        identity_attack_score=category_scores_data["identity_attack"],
        safety_risk_index=inference_result["safety_risk_index"],
        moderation_action=inference_result["moderation_action"],
        engine_version=inference_result["engine_version"],
        processing_time_ms=inference_result["processing_time_ms"],
    )
    session.add(safety_assessment)
    session.commit()

    return MessageAnalysisResponse(
        record_id=content_record.record_id,
        message_body=request.message_body,
        category_scores=SafetyCategoryScores(**category_scores_data),
        triggered_categories=inference_result["triggered_categories"],
        safety_risk_index=inference_result["safety_risk_index"],
        moderation_action=inference_result["moderation_action"],
        engine_version=inference_result["engine_version"],
        processing_time_ms=inference_result["processing_time_ms"],
    )
