"""
CivilityAI: Centralized Application & Model Configuration.

Defines schemas, domain mappings, policy weights, decision cutoffs,
and hardware execution configurations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# Base Workspace Directories
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = WORKSPACE_ROOT / "datasets"
SAVED_MODELS_DIR = WORKSPACE_ROOT / "saved_models"
ANALYSIS_OUTPUTS_DIR = WORKSPACE_ROOT / "analysis_outputs"

# Schema Translation Maps
# Maps raw Kaggle Jigsaw CSV columns to internal domain terminology
RAW_TO_INTERNAL_COLUMN_MAP: Dict[str, str] = {
    "toxic": "general_toxicity",
    "severe_toxic": "severe_abuse",
    "obscene": "obscene_language",
    "threat": "threatening_language",
    "insult": "personal_insult",
    "identity_hate": "identity_attack",
}

INTERNAL_TO_RAW_COLUMN_MAP: Dict[str, str] = {
    internal_name: raw_name
    for raw_name, internal_name in RAW_TO_INTERNAL_COLUMN_MAP.items()
}

SAFETY_CATEGORIES: Dict[str, str] = dict(RAW_TO_INTERNAL_COLUMN_MAP)

# List of all 6 canonical safety category names
CATEGORY_ORDER: List[str] = [
    "general_toxicity",
    "severe_abuse",
    "obscene_language",
    "threatening_language",
    "personal_insult",
    "identity_attack",
]


@dataclass(frozen=True)
class SafetyModelConfig:
    """
    Configuration parameters for Transformer and Baseline architectures.
    """
    model_name: str = "distilbert-base-uncased"
    max_sequence_length: int = 128
    training_batch_size: int = 16
    validation_batch_size: int = 32
    inference_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    training_rounds: int = 3
    early_stopping_patience: int = 2
    dropout_probability: float = 0.20
    warmup_ratio: float = 0.10
    random_seed: int = 42
    device_preference: str = "auto"  # 'auto', 'cuda', or 'cpu'

    baseline_max_features: int = 25000
    baseline_ngram_min: int = 1
    baseline_ngram_max: int = 2


@dataclass(frozen=True)
class ModerationPolicyConfig:
    """
    Policy governance weights and decision boundaries for Content Safety Moderation.
    """
    # Relative severity weights for computing Safety Risk Index (must sum to 1.0)
    category_severity_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "threatening_language": 0.28,
            "severe_abuse": 0.25,
            "identity_attack": 0.22,
            "personal_insult": 0.12,
            "obscene_language": 0.08,
            "general_toxicity": 0.05,
        }
    )

    # Initial decision cutoffs per category (tuned automatically via cutoff_tuning.py)
    decision_cutoffs: Dict[str, float] = field(
        default_factory=lambda: {
            "general_toxicity": 0.50,
            "severe_abuse": 0.35,
            "obscene_language": 0.45,
            "threatening_language": 0.30,
            "personal_insult": 0.45,
            "identity_attack": 0.35,
        }
    )

    # Risk tier thresholds for automated moderation actions
    # Risk < allow_risk_ceiling -> ALLOW
    # allow_risk_ceiling <= Risk < review_risk_ceiling -> REVIEW
    # Risk >= review_risk_ceiling -> ESCALATE
    allow_risk_ceiling: float = 0.30
    review_risk_ceiling: float = 0.75


@dataclass
class ApplicationConfig:
    """
    Central runtime configuration binding models, policies, database, and API.
    """
    environment: str = os.getenv("ENVIRONMENT", "development")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    database_url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{WORKSPACE_ROOT / 'civility_moderation.db'}",
    )
    engine_version: str = "CivilityAI-v1.0.0"
    model_backend: str = os.getenv("MODEL_BACKEND", "transformer")  # 'transformer' or 'baseline'
    
    model_config: SafetyModelConfig = field(default_factory=SafetyModelConfig)
    policy_config: ModerationPolicyConfig = field(default_factory=ModerationPolicyConfig)
