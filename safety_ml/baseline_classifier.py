"""
CivilityAI: Baseline Multi-Label Machine Learning Pipeline.

Implements TF-IDF feature extraction combined with per-category Logistic Regression
classifiers. Provides training, serialization, inference, and rigorous evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from safety_ml.data_pipeline import load_raw_dataset, partition_dataset
from safety_ml.model_assessment import assess_safety_model, format_assessment_table
from safety_ml.settings import (
    CATEGORY_ORDER,
    SAVED_MODELS_DIR,
    SafetyModelConfig,
)

logger = logging.getLogger("CivilityAI.BaselineClassifier")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BaselineSafetyPipeline:
    """
    Multi-label baseline classifier using TF-IDF n-gram vectors and independent
    Logistic Regression models for each safety category.
    """

    def __init__(
        self,
        config: Optional[SafetyModelConfig] = None,
        category_columns: Optional[List[str]] = None,
    ):
        self.config = config or SafetyModelConfig()
        self.category_columns = category_columns or CATEGORY_ORDER

        # Text Feature Extractor: Word + sub-word n-grams
        self.text_feature_extractor = TfidfVectorizer(
            ngram_range=(self.config.baseline_ngram_min, self.config.baseline_ngram_max),
            max_features=self.config.baseline_max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )

        # One binary LogisticRegression classifier per safety category with class_weight='balanced'
        self.category_classifiers: Dict[str, LogisticRegression] = {
            category: LogisticRegression(
                C=2.0,
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear",
                random_state=self.config.random_seed,
            )
            for category in self.category_columns
        }

        self.is_fitted = False

    def train(
        self,
        training_messages: List[str],
        training_targets: np.ndarray,
    ) -> "BaselineSafetyPipeline":
        """
        Fits the TF-IDF extractor on training text and trains individual category classifiers.
        """
        logger.info(f"Extracting TF-IDF features for {len(training_messages)} training samples...")
        training_features = self.text_feature_extractor.fit_transform(training_messages)

        logger.info("Fitting per-category Logistic Regression classifiers...")
        for idx, category in enumerate(self.category_columns):
            category_target = training_targets[:, idx]
            # Check if there is variation in target
            if len(np.unique(category_target)) > 1:
                self.category_classifiers[category].fit(training_features, category_target)
            else:
                logger.warning(f"Category '{category}' has only 1 class in training partition.")

        self.is_fitted = True
        logger.info("Baseline safety pipeline fitting complete.")
        return self

    def predict_risk_probabilities(
        self,
        messages: List[str],
    ) -> np.ndarray:
        """
        Infers continuous risk probabilities for each safety category across input messages.
        Returns array of shape (N, num_categories).
        """
        if not self.is_fitted:
            raise RuntimeError("BaselineSafetyPipeline must be trained or loaded before prediction.")

        features = self.text_feature_extractor.transform(messages)
        probability_matrix = np.zeros((len(messages), len(self.category_columns)), dtype=np.float32)

        for idx, category in enumerate(self.category_columns):
            classifier = self.category_classifiers[category]
            if hasattr(classifier, "classes_") and len(classifier.classes_) > 1:
                # Column 1 corresponds to probability of positive label
                probability_matrix[:, idx] = classifier.predict_proba(features)[:, 1]
            else:
                probability_matrix[:, idx] = 0.0

        return probability_matrix

    def save_artifacts(self, destination_dir: Optional[Path | str] = None) -> Path:
        """
        Serializes the TF-IDF vectorizer and trained classifiers.
        """
        if destination_dir is None:
            destination_dir = SAVED_MODELS_DIR / "baseline"

        target_dir = Path(destination_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        feature_extractor_path = target_dir / "text_feature_extractor.joblib"
        classifiers_path = target_dir / "category_classifiers.joblib"

        joblib.dump(self.text_feature_extractor, feature_extractor_path)
        joblib.dump(self.category_classifiers, classifiers_path)

        logger.info(f"Baseline artifacts successfully saved under: {target_dir.resolve()}")
        return target_dir

    @classmethod
    def load_artifacts(cls, source_dir: Optional[Path | str] = None) -> "BaselineSafetyPipeline":
        """
        Loads serialized TF-IDF vectorizer and classifiers from disk.
        """
        if source_dir is None:
            source_dir = SAVED_MODELS_DIR / "baseline"

        source_path = Path(source_dir)
        pipeline = cls()
        pipeline.text_feature_extractor = joblib.load(source_path / "text_feature_extractor.joblib")
        pipeline.category_classifiers = joblib.load(source_path / "category_classifiers.joblib")
        pipeline.is_fitted = True
        logger.info(f"Loaded baseline artifacts from: {source_path.resolve()}")
        return pipeline


def execute_baseline_training_run(
    source_dataset_path: Optional[Path | str] = None,
) -> Tuple[BaselineSafetyPipeline, Dict[str, Any]]:
    """
    Orchestrates end-to-end dataset loading, splitting, baseline model training,
    artifact persistence, and multi-label validation assessment.
    """
    message_frame = load_raw_dataset(source_dataset_path)
    training_frame, validation_frame, evaluation_frame = partition_dataset(message_frame)

    category_columns = CATEGORY_ORDER
    training_messages = training_frame["message_body"].tolist()
    training_targets = training_frame[category_columns].values

    validation_messages = validation_frame["message_body"].tolist()
    validation_targets = validation_frame[category_columns].values

    # Initialize and train
    pipeline = BaselineSafetyPipeline(category_columns=category_columns)
    pipeline.train(training_messages, training_targets)

    # Evaluate on validation partition
    logger.info("Evaluating baseline model on validation partition...")
    validation_risk_probabilities = pipeline.predict_risk_probabilities(validation_messages)
    assessment_report = assess_safety_model(
        true_targets=validation_targets,
        risk_probabilities=validation_risk_probabilities,
        category_columns=category_columns,
    )

    table_repr = format_assessment_table(assessment_report)
    print("\n" + "=" * 90)
    print("BASELINE ML MODEL: VALIDATION EVALUATION REPORT (TF-IDF + Logistic Regression)")
    print("=" * 90)
    print(table_repr)
    print("=" * 90 + "\n")

    # Persist artifacts
    pipeline.save_artifacts()

    return pipeline, assessment_report


if __name__ == "__main__":
    execute_baseline_training_run()
