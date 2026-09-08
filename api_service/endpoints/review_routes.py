"""
CivilityAI: Human Moderator Review Console Endpoints.
"""

from __future__ import annotations

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from api_service.contracts.schemas import (
    ModeratorActionRequest,
    PendingReviewItem,
)
from api_service.entities.models import (
    ContentRecord,
    ReviewOutcome,
    SafetyAssessment,
)
from api_service.persistence import get_database_session

logger = logging.getLogger("CivilityAI.ReviewRoutes")
review_router = APIRouter(prefix="/review", tags=["Review Console"])


@review_router.get("/pending", response_model=List[PendingReviewItem])
def get_pending_review_queue(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_database_session),
) -> List[PendingReviewItem]:
    """
    Retrieves messages flagged for human review ('REVIEW' or 'ESCALATE') that have
    not yet been resolved by a human moderator.
    """
    # Find assessments requiring human attention that do not have a finalized ReviewOutcome
    query = (
        session.query(ContentRecord, SafetyAssessment)
        .join(SafetyAssessment, ContentRecord.record_id == SafetyAssessment.content_record_id)
        .outerjoin(ReviewOutcome, ContentRecord.record_id == ReviewOutcome.content_record_id)
        .filter(
            SafetyAssessment.moderation_action.in_(["REVIEW", "ESCALATE"]),
            ReviewOutcome.review_id.is_(None),  # Unresolved only
        )
        .order_by(SafetyAssessment.safety_risk_index.desc(), ContentRecord.submitted_at.desc())
        .offset(offset)
        .limit(limit)
    )

    results = query.all()
    queue_items: List[PendingReviewItem] = []

    for record, assessment in results:
        scores = {
            "general_toxicity": assessment.general_toxicity_score,
            "severe_abuse": assessment.severe_abuse_score,
            "obscene_language": assessment.obscene_language_score,
            "threatening_language": assessment.threatening_language_score,
            "personal_insult": assessment.personal_insult_score,
            "identity_attack": assessment.identity_attack_score,
        }
        # Determine highest scoring category
        primary_cat = max(scores.items(), key=lambda item: item[1])[0]

        queue_items.append(
            PendingReviewItem(
                record_id=record.record_id,
                message_body=record.message_body,
                submitted_at=record.submitted_at,
                safety_risk_index=assessment.safety_risk_index,
                moderation_action=assessment.moderation_action,
                primary_category=primary_cat,
                category_scores=scores,
            )
        )

    return queue_items


@review_router.post("/{record_id}/action", status_code=status.HTTP_200_OK)
def submit_moderator_action(
    record_id: str,
    action_request: ModeratorActionRequest,
    session: Session = Depends(get_database_session),
):
    """
    Records a human moderator's resolution (ALLOW, REMOVE, ESCALATE) on a flagged message.
    """
    record = session.query(ContentRecord).filter(ContentRecord.record_id == record_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Content record with ID '{record_id}' not found.",
        )

    review_outcome = ReviewOutcome(
        content_record_id=record.record_id,
        review_action=action_request.review_action,
        reviewer_identifier=action_request.reviewer_identifier,
        notes=action_request.notes,
    )
    session.add(review_outcome)
    session.commit()

    logger.info(
        f"Moderator '{action_request.reviewer_identifier}' applied '{action_request.review_action}' "
        f"to record {record_id}."
    )

    return {
        "status": "success",
        "record_id": record_id,
        "review_action": action_request.review_action,
        "reviewer": action_request.reviewer_identifier,
    }
