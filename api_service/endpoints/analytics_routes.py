"""
CivilityAI: Analytics & Platform Monitoring Endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api_service.contracts.schemas import AnalyticsSummaryResponse
from api_service.entities.models import ContentRecord, SafetyAssessment
from api_service.persistence import get_database_session
from safety_ml.settings import CATEGORY_ORDER

analytics_router = APIRouter(prefix="/analytics", tags=["Analytics & Monitoring"])


@analytics_router.get("/summary", response_model=AnalyticsSummaryResponse)
def get_analytics_summary(
    session: Session = Depends(get_database_session),
) -> AnalyticsSummaryResponse:
    """
    Computes real-time platform KPIs, action distributions, category frequencies,
    and inference latency statistics (Mean and P95).
    """
    assessments: List[SafetyAssessment] = session.query(SafetyAssessment).all()
    total_analyzed = len(assessments)

    if total_analyzed == 0:
        return AnalyticsSummaryResponse(
            total_analyzed=0,
            allowed_count=0,
            review_count=0,
            escalated_count=0,
            allowed_percentage=0.0,
            review_percentage=0.0,
            escalated_percentage=0.0,
            average_processing_time_ms=0.0,
            p95_processing_time_ms=0.0,
            category_distribution={cat: 0 for cat in CATEGORY_ORDER},
            recent_high_risk_content=[],
        )

    allowed_count = sum(1 for a in assessments if a.moderation_action == "ALLOW")
    review_count = sum(1 for a in assessments if a.moderation_action == "REVIEW")
    escalated_count = sum(1 for a in assessments if a.moderation_action == "ESCALATE")

    allowed_pct = round((allowed_count / total_analyzed) * 100.0, 2)
    review_pct = round((review_count / total_analyzed) * 100.0, 2)
    escalated_pct = round((escalated_count / total_analyzed) * 100.0, 2)

    latencies = [a.processing_time_ms for a in assessments]
    avg_latency = round(float(np.mean(latencies)), 2)
    p95_latency = round(float(np.percentile(latencies, 95)), 2)

    # Category triggers (score >= 0.50)
    category_distribution = {
        "general_toxicity": sum(1 for a in assessments if a.general_toxicity_score >= 0.50),
        "severe_abuse": sum(1 for a in assessments if a.severe_abuse_score >= 0.35),
        "obscene_language": sum(1 for a in assessments if a.obscene_language_score >= 0.45),
        "threatening_language": sum(1 for a in assessments if a.threatening_language_score >= 0.30),
        "personal_insult": sum(1 for a in assessments if a.personal_insult_score >= 0.45),
        "identity_attack": sum(1 for a in assessments if a.identity_attack_score >= 0.35),
    }

    # Fetch 5 most recent high risk items
    high_risk_query = (
        session.query(ContentRecord, SafetyAssessment)
        .join(SafetyAssessment, ContentRecord.record_id == SafetyAssessment.content_record_id)
        .filter(SafetyAssessment.safety_risk_index >= 0.70)
        .order_by(SafetyAssessment.assessed_at.desc())
        .limit(5)
        .all()
    )

    recent_high_risk: List[Dict[str, Any]] = [
        {
            "record_id": record.record_id,
            "message_body": record.message_body,
            "safety_risk_index": assessment.safety_risk_index,
            "moderation_action": assessment.moderation_action,
            "assessed_at": assessment.assessed_at.isoformat(),
        }
        for record, assessment in high_risk_query
    ]

    return AnalyticsSummaryResponse(
        total_analyzed=total_analyzed,
        allowed_count=allowed_count,
        review_count=review_count,
        escalated_count=escalated_count,
        allowed_percentage=allowed_pct,
        review_percentage=review_pct,
        escalated_percentage=escalated_pct,
        average_processing_time_ms=avg_latency,
        p95_processing_time_ms=p95_latency,
        category_distribution=category_distribution,
        recent_high_risk_content=recent_high_risk,
    )
