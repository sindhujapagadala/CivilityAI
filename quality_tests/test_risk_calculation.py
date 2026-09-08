"""
CivilityAI: Unit Tests for Safety Risk Index & Policy Decisions.
"""

import pytest
from safety_ml.safety_inference import compute_safety_risk
from safety_ml.settings import ModerationPolicyConfig


def test_safe_content_receives_allow_action():
    """Verifies that completely safe content receives a low risk index and ALLOW."""
    zero_scores = {
        "general_toxicity": 0.02,
        "severe_abuse": 0.01,
        "obscene_language": 0.02,
        "threatening_language": 0.01,
        "personal_insult": 0.01,
        "identity_attack": 0.01,
    }
    risk, action = compute_safety_risk(zero_scores)
    assert risk < 0.30
    assert action == "ALLOW"


def test_moderate_violation_triggers_review_action():
    """Verifies that borderline insults trigger HUMAN REVIEW action."""
    moderate_scores = {
        "general_toxicity": 0.65,
        "severe_abuse": 0.10,
        "obscene_language": 0.30,
        "threatening_language": 0.05,
        "personal_insult": 0.60,
        "identity_attack": 0.05,
    }
    risk, action = compute_safety_risk(moderate_scores)
    assert 0.30 <= risk < 0.75
    assert action == "REVIEW"


def test_high_threat_triggers_escalate_and_severe_amplification():
    """
    Verifies that a high threat score cannot be diluted by zeros in other categories,
    activating the critical category non-linear amplification and triggering ESCALATE.
    """
    threat_scores = {
        "general_toxicity": 0.10,
        "severe_abuse": 0.05,
        "obscene_language": 0.05,
        "threatening_language": 0.95,
        "personal_insult": 0.10,
        "identity_attack": 0.05,
    }
    risk, action = compute_safety_risk(threat_scores)
    assert risk >= 0.75
    assert action == "ESCALATE"


def test_custom_policy_thresholds():
    """Verifies custom moderation boundary configuration."""
    custom_policy = ModerationPolicyConfig(
        allow_risk_ceiling=0.20,
        review_risk_ceiling=0.60,
    )
    scores = {
        "general_toxicity": 0.40,
        "severe_abuse": 0.20,
        "obscene_language": 0.20,
        "threatening_language": 0.20,
        "personal_insult": 0.30,
        "identity_attack": 0.10,
    }
    risk, action = compute_safety_risk(scores, policy_config=custom_policy)
    assert 0.20 <= risk < 0.60
    assert action == "REVIEW"


def test_risk_score_bounds_are_strictly_bounded():
    """Verifies risk index never exceeds 1.0 or falls below 0.0 under extreme inputs."""
    max_scores = {cat: 1.0 for cat in ["general_toxicity", "severe_abuse", "obscene_language", "threatening_language", "personal_insult", "identity_attack"]}
    risk_max, action_max = compute_safety_risk(max_scores)
    assert risk_max <= 1.0
    assert action_max == "ESCALATE"

    min_scores = {cat: 0.0 for cat in ["general_toxicity", "severe_abuse", "obscene_language", "threatening_language", "personal_insult", "identity_attack"]}
    risk_min, action_min = compute_safety_risk(min_scores)
    assert risk_min == 0.0
    assert action_min == "ALLOW"
