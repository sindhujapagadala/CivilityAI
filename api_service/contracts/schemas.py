"""
CivilityAI: Pydantic Data Contracts & API Schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SafetyCategoryScores(BaseModel):
    """
    Independent probabilities [0.0 - 1.0] for all 6 safety categories.
    """
    general_toxicity: float = Field(..., ge=0.0, le=1.0, description="Probability of general toxicity")
    severe_abuse: float = Field(..., ge=0.0, le=1.0, description="Probability of severe toxicity / abuse")
    obscene_language: float = Field(..., ge=0.0, le=1.0, description="Probability of profanity or obscene language")
    threatening_language: float = Field(..., ge=0.0, le=1.0, description="Probability of physical threat or violence")
    personal_insult: float = Field(..., ge=0.0, le=1.0, description="Probability of personal insults or disparagement")
    identity_attack: float = Field(..., ge=0.0, le=1.0, description="Probability of identity hate or bigotry")


class MessageAnalysisRequest(BaseModel):
    """
    Incoming user payload for safety assessment.
    """
    message_body: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text content to be assessed for safety violations",
        example="You are completely useless.",
    )

    @field_validator("message_body")
    @classmethod
    def validate_message_not_empty_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message body cannot consist exclusively of whitespace.")
        return value


class MessageAnalysisResponse(BaseModel):
    """
    Comprehensive safety assessment returned by the engine.
    """
    record_id: Optional[str] = Field(None, description="Unique persistence identifier for this record")
    message_body: str = Field(..., description="Original user message")
    category_scores: SafetyCategoryScores
    triggered_categories: List[str] = Field(default_factory=list, description="Categories exceeding decision cutoffs")
    safety_risk_index: float = Field(..., ge=0.0, le=1.0, description="Weighted composite risk index [0.0, 1.0]")
    moderation_action: str = Field(..., description="Automated decision: ALLOW, REVIEW, or ESCALATE")
    engine_version: str = Field(..., description="Safety engine version identifier")
    processing_time_ms: float = Field(..., description="Inference execution latency in milliseconds")


class ModeratorActionRequest(BaseModel):
    """
    Action submitted by human moderator on flagged content.
    """
    review_action: str = Field(
        ...,
        description="Human resolution action: ALLOW, REMOVE, or ESCALATE",
        example="REMOVE",
    )
    reviewer_identifier: str = Field(
        default="human_reviewer_1",
        description="Identifier of the reviewer",
    )
    notes: Optional[str] = Field(
        None,
        description="Optional moderator notes",
    )

    @field_validator("review_action")
    @classmethod
    def validate_action(cls, value: str) -> str:
        upper = value.upper().strip()
        if upper not in {"ALLOW", "REMOVE", "ESCALATE"}:
            raise ValueError("review_action must be one of: 'ALLOW', 'REMOVE', 'ESCALATE'")
        return upper


class PendingReviewItem(BaseModel):
    """
    Content record flagged for human moderator attention.
    """
    record_id: str
    message_body: str
    submitted_at: datetime
    safety_risk_index: float
    moderation_action: str
    primary_category: str
    category_scores: Dict[str, float]


class AnalyticsSummaryResponse(BaseModel):
    """
    Aggregated operational statistics for platform monitoring.
    """
    total_analyzed: int
    allowed_count: int
    review_count: int
    escalated_count: int
    allowed_percentage: float
    review_percentage: float
    escalated_percentage: float
    average_processing_time_ms: float
    p95_processing_time_ms: float
    category_distribution: Dict[str, int]
    recent_high_risk_content: List[Dict[str, Any]]


class SystemHealthResponse(BaseModel):
    """
    Service health and runtime configuration check.
    """
    status: str
    engine_version: str
    model_backend: str
    device: str
    database_connected: bool
    timestamp: datetime
