"""
CivilityAI: Unit Tests for Inference Engine & Singleton Lifecycle.
"""

import pytest
from safety_ml.safety_inference import ContentSafetyEngine


def test_engine_is_not_reloaded_per_request():
    """
    CRITICAL REQUIREMENT: Verifies that ContentSafetyEngine follows the singleton
    pattern and is not re-instantiated or reloaded on subsequent requests.
    """
    instance_one = ContentSafetyEngine.get_singleton_instance()
    instance_two = ContentSafetyEngine.get_singleton_instance()
    assert instance_one is instance_two
    assert id(instance_one) == id(instance_two)


def test_analyze_message_civil_content():
    """Verifies that normal civil text receives ALLOW and low category scores."""
    engine = ContentSafetyEngine.get_singleton_instance()
    result = engine.analyze_message("Thank you for contributing reliable sources to this article.")

    assert result["moderation_action"] == "ALLOW"
    assert result["safety_risk_index"] < 0.30
    assert len(result["triggered_categories"]) == 0
    assert "general_toxicity" in result["category_scores"]
    assert result["processing_time_ms"] >= 0


def test_analyze_message_toxic_content():
    """Verifies that an abusive insult triggers personal_insult and non-ALLOW action."""
    engine = ContentSafetyEngine.get_singleton_instance()
    result = engine.analyze_message("You are completely incompetent and an absolute idiot.")

    assert result["moderation_action"] in ["REVIEW", "ESCALATE"]
    assert result["safety_risk_index"] >= 0.30
    assert "personal_insult" in result["category_scores"]
    assert result["category_scores"]["personal_insult"] > 0.50


def test_analyze_batch_processing():
    """Verifies batched inference over multiple messages."""
    engine = ContentSafetyEngine.get_singleton_instance()
    messages = [
        "Great work today!",
        "You are an idiot.",
        "Let us discuss on the talk page.",
    ]
    results = engine.analyze_batch(messages)
    assert len(results) == 3
    assert results[0]["moderation_action"] == "ALLOW"
    assert results[1]["moderation_action"] in ["REVIEW", "ESCALATE"]
    assert results[2]["moderation_action"] == "ALLOW"


def test_empty_message_is_safe_by_default():
    """Verifies that empty string or whitespace does not crash and returns 0 risk."""
    engine = ContentSafetyEngine.get_singleton_instance()
    result = engine.analyze_message("    ")
    assert result["safety_risk_index"] == 0.0
    assert result["moderation_action"] == "ALLOW"
    assert len(result["triggered_categories"]) == 0
