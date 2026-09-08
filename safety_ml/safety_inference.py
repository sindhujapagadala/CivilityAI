"""
CivilityAI: Content Safety Inference Engine.

Provides low-latency, thread-safe inference across safety categories, calculates
the aggregate Safety Risk Index with calibrated policy weights, and determines
automated moderation actions (ALLOW, REVIEW, ESCALATE).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from safety_ml.cutoff_tuning import load_decision_cutoffs
from safety_ml.settings import (
    CATEGORY_ORDER,
    SAVED_MODELS_DIR,
    ApplicationConfig,
    ModerationPolicyConfig,
    SafetyModelConfig,
)
from safety_ml.text_processing import prepare_message_text

logger = logging.getLogger("CivilityAI.SafetyInference")


def compute_safety_risk(
    category_scores: Dict[str, float],
    policy_config: Optional[ModerationPolicyConfig] = None,
) -> Tuple[float, str]:
    """
    Computes the composite Safety Risk Index and determines the appropriate moderation action.

    Formula:
        Safety Risk Index = sum(category_score * category_severity_weight)
        With a non-linear amplification if critical severe categories (Threat, Severe Abuse,
        Identity Attack) exceed high alarm thresholds.

    Moderation Actions:
        Risk < ALLOW_RISK_CEILING (default 0.30)           -> ALLOW
        ALLOW_RISK_CEILING <= Risk < REVIEW_RISK_CEILING   -> REVIEW
        Risk >= REVIEW_RISK_CEILING (default 0.75)         -> ESCALATE
    """
    if policy_config is None:
        policy_config = ModerationPolicyConfig()

    weights = policy_config.category_severity_weights
    weighted_score = 0.0

    for category, score in category_scores.items():
        weight = weights.get(category, 0.10)
        weighted_score += score * weight

    # Severe category amplification: severe abuse or threats cannot be diluted by low scores in other categories
    critical_categories = ["threatening_language", "severe_abuse", "identity_attack"]
    max_critical_score = max((category_scores.get(cat, 0.0) for cat in critical_categories), default=0.0)

    # If any critical violent or hate category exceeds 0.70, risk is at least 85% of that critical score
    if max_critical_score > 0.70:
        weighted_score = max(weighted_score, max_critical_score * 0.90)

    safety_risk_index = round(float(min(max(weighted_score, 0.0), 1.0)), 4)

    # Determine automated moderation action
    if safety_risk_index < policy_config.allow_risk_ceiling:
        moderation_action = "ALLOW"
    elif safety_risk_index < policy_config.review_risk_ceiling:
        moderation_action = "REVIEW"
    else:
        moderation_action = "ESCALATE"

    return safety_risk_index, moderation_action


class ContentSafetyEngine:
    """
    Production-grade Inference Engine.
    Loads models once at initialization and reuses them across requests.
    Supports DistilBERT transformer, baseline fallback, and fast batch processing.
    """

    _instance: Optional["ContentSafetyEngine"] = None

    def __init__(
        self,
        app_config: Optional[ApplicationConfig] = None,
        model_directory: Optional[Path | str] = None,
    ):
        self.app_config = app_config or ApplicationConfig()
        self.model_config = self.app_config.model_config
        self.policy_config = self.app_config.policy_config
        self.category_columns = CATEGORY_ORDER
        self.engine_version = self.app_config.engine_version

        self.model_directory = Path(model_directory or (SAVED_MODELS_DIR / "transformer"))
        self.decision_cutoffs = load_decision_cutoffs()

        # Determine execution hardware
        if torch.cuda.is_available() and self.model_config.device_preference != "cpu":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.language_guard = None
        self.text_encoder = None
        self.baseline_pipeline = None
        self.is_loaded = False

        self._initialize_underlying_model()

    @classmethod
    def get_singleton_instance(
        cls,
        app_config: Optional[ApplicationConfig] = None,
    ) -> "ContentSafetyEngine":
        """
        Ensures the model is loaded exactly once into application memory.
        """
        if cls._instance is None:
            logger.info("Instantiating ContentSafetyEngine singleton...")
            cls._instance = cls(app_config=app_config)
        return cls._instance

    def _initialize_underlying_model(self) -> None:
        """
        Loads the configured model architecture into memory.
        Attempts transformer loading first; falls back gracefully if weights are not yet trained.
        """
        weights_file = self.model_directory / "safety_guard_weights.pt"

        if self.app_config.model_backend == "transformer" and weights_file.exists():
            try:
                from transformers import AutoTokenizer
                from safety_ml.transformer_classifier import ContentSafetyTransformer

                logger.info(f"Loading transformer tokenizer from {self.model_directory}...")
                self.text_encoder = AutoTokenizer.from_pretrained(self.model_directory)

                logger.info(f"Loading transformer weights from {weights_file}...")
                self.language_guard = ContentSafetyTransformer(
                    base_model_identifier=self.model_config.model_name,
                    num_safety_categories=len(self.category_columns),
                )
                self.language_guard.load_state_dict(torch.load(weights_file, map_location=self.device))
                self.language_guard.to(self.device)
                self.language_guard.eval()
                self.is_loaded = True
                logger.info(f"Transformer model active on {self.device}.")
                return
            except Exception as exc:
                logger.warning(f"Failed loading trained transformer ({exc}). Falling back to baseline.")

        # Check baseline artifacts
        baseline_dir = SAVED_MODELS_DIR / "baseline"
        if (baseline_dir / "text_feature_extractor.joblib").exists():
            try:
                from safety_ml.baseline_classifier import BaselineSafetyPipeline

                logger.info(f"Loading baseline ML pipeline from {baseline_dir}...")
                self.baseline_pipeline = BaselineSafetyPipeline.load_artifacts(baseline_dir)
                self.is_loaded = True
                logger.info("Baseline ML pipeline loaded successfully.")
                return
            except Exception as exc:
                logger.warning(f"Failed loading baseline model ({exc}).")

        # Fallback / Lightweight heuristic safety mode for instant development and testing
        logger.info("Using embedded lightweight heuristic safety guard (pre-training mode).")
        self.is_loaded = True

    def analyze_message(
        self,
        raw_message: str,
    ) -> Dict[str, Any]:
        """
        Main inference entrypoint.
        Cleans text, executes inference, applies cutoffs, and computes Safety Risk Index.

        Returns:
            Structured dictionary matching MessageAnalysisResponse schema.
        """
        start_time = time.perf_counter()

        sanitized_message = prepare_message_text(raw_message)

        if not sanitized_message.strip():
            # Empty or whitespace only content is safe by default
            processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "sanitized_message": "",
                "category_scores": {cat: 0.0 for cat in self.category_columns},
                "triggered_categories": [],
                "safety_risk_index": 0.0,
                "moderation_action": "ALLOW",
                "engine_version": self.engine_version,
                "processing_time_ms": processing_time_ms,
            }

        # 1. Obtain raw probabilities
        category_scores: Dict[str, float] = {}

        if self.language_guard is not None and self.text_encoder is not None:
            encoding = self.text_encoder(
                sanitized_message,
                truncation=True,
                max_length=self.model_config.max_sequence_length,
                padding="max_length",
                return_tensors="pt",
            )
            encoded_tokens = encoding["input_ids"].to(self.device)
            token_visibility = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.language_guard(encoded_tokens, token_visibility)
                probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            for idx, category in enumerate(self.category_columns):
                category_scores[category] = round(float(probabilities[idx]), 4)

        elif self.baseline_pipeline is not None:
            probs = self.baseline_pipeline.predict_risk_probabilities([sanitized_message])[0]
            for idx, category in enumerate(self.category_columns):
                category_scores[category] = round(float(probs[idx]), 4)

        else:
            # Embedded rule-based heuristic guard for pre-training testing
            category_scores = self._heuristic_scoring(sanitized_message)

        # 2. Determine triggered categories using decision cutoffs
        triggered_categories: List[str] = [
            category
            for category, score in category_scores.items()
            if score >= self.decision_cutoffs.get(category, 0.50)
        ]

        # 3. Compute Safety Risk Index and Moderation Action
        safety_risk_index, moderation_action = compute_safety_risk(
            category_scores=category_scores,
            policy_config=self.policy_config,
        )

        processing_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "sanitized_message": sanitized_message,
            "category_scores": category_scores,
            "triggered_categories": triggered_categories,
            "safety_risk_index": safety_risk_index,
            "moderation_action": moderation_action,
            "engine_version": self.engine_version,
            "processing_time_ms": processing_time_ms,
        }

    def analyze_batch(
        self,
        messages: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Analyzes a collection of messages efficiently.
        """
        return [self.analyze_message(msg) for msg in messages]

    def _heuristic_scoring(self, text: str) -> Dict[str, float]:
        """
        Deterministic pattern matcher used when weights are not yet persisted to disk.
        """
        lower = text.lower()
        scores = {cat: 0.02 for cat in self.category_columns}

        # Threat patterns
        threat_terms = ["kill", "die", "break your", "shoot", "bullet", "hunt you", "murder", "beat you"]
        if any(term in lower for term in threat_terms):
            scores["threatening_language"] = 0.94
            scores["severe_abuse"] = 0.85
            scores["general_toxicity"] = 0.92

        # Severe abuse
        abuse_terms = ["suffer and die", "rot in", "destroy you", "worthless piece of human"]
        if any(term in lower for term in abuse_terms):
            scores["severe_abuse"] = 0.91
            scores["general_toxicity"] = 0.95

        # Obscenity
        obscene_terms = ["fuck", "shit", "bitch", "crap", "asshole", "bullshit"]
        if any(term in lower for term in obscene_terms):
            scores["obscene_language"] = 0.93
            scores["general_toxicity"] = max(scores["general_toxicity"], 0.82)

        # Insult
        insult_terms = ["idiot", "moron", "loser", "stupid", "incompetent", "garbage", "trash", "clown"]
        if any(term in lower for term in insult_terms):
            scores["personal_insult"] = 0.92
            scores["general_toxicity"] = max(scores["general_toxicity"], 0.88)

        # Identity hate
        hate_terms = ["parasite", "subhuman", "your kind", "ethnic", "religion is a disease", "wiped out"]
        if any(term in lower for term in hate_terms):
            scores["identity_attack"] = 0.93
            scores["severe_abuse"] = max(scores["severe_abuse"], 0.80)
            scores["general_toxicity"] = max(scores["general_toxicity"], 0.94)

        return {k: round(v, 4) for k, v in scores.items()}
