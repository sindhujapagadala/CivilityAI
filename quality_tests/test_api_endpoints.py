"""
CivilityAI: Integration Tests for FastAPI REST Endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from api_service.application import api_application
from api_service.persistence import init_database

# Initialize database schema and test client
init_database()
client = TestClient(api_application)


def test_system_health_endpoint():
    """Verifies GET /system/health reports operational status."""
    response = client.get("/system/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "engine_version" in data
    assert data["database_connected"] is True


def test_safety_analyze_endpoint_civil():
    """Verifies POST /safety/analyze returns ALLOW for civil message."""
    payload = {"message_body": "Thank you for the detailed and constructive review."}
    response = client.post("/safety/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["moderation_action"] == "ALLOW"
    assert data["safety_risk_index"] < 0.30
    assert "record_id" in data
    assert "category_scores" in data
    assert "processing_time_ms" in data


def test_safety_analyze_endpoint_threat():
    """Verifies POST /safety/analyze detects threat and returns non-ALLOW action."""
    payload = {"message_body": "I will find where you live and break your skull."}
    response = client.post("/safety/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["moderation_action"] in ["REVIEW", "ESCALATE"]
    assert data["safety_risk_index"] >= 0.70
    assert data["category_scores"]["threatening_language"] > 0.50


def test_empty_message_is_rejected_by_api_validation():
    """Verifies that empty/whitespace message fails Pydantic schema validation with 422."""
    response = client.post("/safety/analyze", json={"message_body": "   "})
    assert response.status_code == 422


def test_missing_message_body_is_rejected():
    """Verifies missing payload field returns 422."""
    response = client.post("/safety/analyze", json={})
    assert response.status_code == 422


def test_review_endpoints_and_workflow():
    """Verifies review queue retrieval and moderator action submission."""
    # 1. First trigger a violation to populate queue
    client.post("/safety/analyze", json={"message_body": "You are a complete idiot."})

    # 2. Fetch pending reviews
    queue_resp = client.get("/review/pending")
    assert queue_resp.status_code == 200
    queue = queue_resp.json()
    assert isinstance(queue, list)

    if len(queue) > 0:
        target_record = queue[0]
        rec_id = target_record["record_id"]

        # 3. Submit moderator action
        action_resp = client.post(
            f"/review/{rec_id}/action",
            json={"review_action": "REMOVE", "reviewer_identifier": "test_moderator_42"},
        )
        assert action_resp.status_code == 200
        action_data = action_resp.json()
        assert action_data["status"] == "success"
        assert action_data["review_action"] == "REMOVE"


def test_analytics_summary_endpoint():
    """Verifies GET /analytics/summary calculates real platform metrics."""
    response = client.get("/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_analyzed" in data
    assert "allowed_count" in data
    assert "average_processing_time_ms" in data
    assert "category_distribution" in data
