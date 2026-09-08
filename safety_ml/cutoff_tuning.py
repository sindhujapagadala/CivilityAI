"""
CivilityAI: Threshold & Decision Cutoff Optimization.

Tunes per-category probability thresholds on validation sets to maximize F1 score
or achieve calibrated Precision/Recall tradeoffs under severe label imbalance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from safety_ml.settings import CATEGORY_ORDER, SAVED_MODELS_DIR

logger = logging.getLogger("CivilityAI.CutoffTuning")


def tune_decision_cutoffs(
    validation_targets: np.ndarray,
    validation_probabilities: np.ndarray,
    category_columns: Optional[List[str]] = None,
    candidate_cutoffs: Optional[np.ndarray] = None,
    target_metric: str = "f1",
    minimum_precision: float = 0.50,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Evaluates candidate thresholds across validation predictions to identify
    the optimal cutoff per category.

    Args:
        validation_targets: Ground truth binary array of shape (N, num_categories).
        validation_probabilities: Model output probabilities of shape (N, num_categories).
        category_columns: Category names.
        candidate_cutoffs: Grid of threshold values to test (defaults to 0.10 -> 0.85 in steps of 0.02).
        target_metric: Metric to maximize ('f1' or 'precision_constrained_recall').
        minimum_precision: Precision floor when using precision_constrained_recall.

    Returns:
        Tuple containing:
        - category_cutoffs: Dict mapping category name to optimal threshold
        - tuning_history: Detailed metrics for all evaluated thresholds
    """
    if category_columns is None:
        category_columns = CATEGORY_ORDER

    if candidate_cutoffs is None:
        candidate_cutoffs = np.arange(0.10, 0.86, 0.02)

    category_cutoffs: Dict[str, float] = {}
    tuning_history: Dict[str, List[Dict[str, float]]] = {}

    for col_idx, category in enumerate(category_columns):
        true_labels = validation_targets[:, col_idx]
        pred_probs = validation_probabilities[:, col_idx]

        # If positive support is 0 in validation partition, use conservative default
        if np.sum(true_labels) == 0:
            category_cutoffs[category] = 0.50
            tuning_history[category] = []
            continue

        best_cutoff = 0.50
        best_score = -1.0
        best_precision = 0.0
        best_recall = 0.0
        category_records: List[Dict[str, float]] = []

        for cutoff in candidate_cutoffs:
            binary_preds = (pred_probs >= cutoff).astype(int)
            prec = float(precision_score(true_labels, binary_preds, zero_division=0))
            rec = float(recall_score(true_labels, binary_preds, zero_division=0))
            f1 = float(f1_score(true_labels, binary_preds, zero_division=0))

            category_records.append({
                "cutoff": float(cutoff),
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
            })

            if target_metric == "f1":
                current_score = f1
            elif target_metric == "precision_constrained_recall":
                # Prefer highest recall while maintaining minimum acceptable precision
                if prec >= minimum_precision:
                    current_score = rec
                else:
                    current_score = prec * 0.1  # Heavy penalty
            else:
                current_score = f1

            if current_score > best_score:
                best_score = current_score
                best_cutoff = float(cutoff)
                best_precision = prec
                best_recall = rec

        # Safety fallback: if best cutoff results in 0 predictions or degenerate score, set standard fallback
        if best_score <= 0.0:
            best_cutoff = 0.40

        category_cutoffs[category] = round(best_cutoff, 3)
        tuning_history[category] = category_records

        logger.info(
            f"Optimized cutoff for '{category}': {best_cutoff:.3f} "
            f"(F1={best_score:.4f}, Precision={best_precision:.4f}, Recall={best_recall:.4f})"
        )

    return category_cutoffs, tuning_history


def save_decision_cutoffs(
    category_cutoffs: Dict[str, float],
    destination_path: Optional[Path | str] = None,
) -> Path:
    """
    Saves the tuned cutoffs to JSON configuration file.
    """
    if destination_path is None:
        destination_path = SAVED_MODELS_DIR / "decision_cutoffs.json"

    dest = Path(destination_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(category_cutoffs, f, indent=2)

    logger.info(f"Saved decision cutoffs to: {dest.resolve()}")
    return dest


def load_decision_cutoffs(source_path: Optional[Path | str] = None) -> Dict[str, float]:
    """
    Loads saved decision cutoffs from file or returns default policy cutoffs.
    """
    if source_path is None:
        source_path = SAVED_MODELS_DIR / "decision_cutoffs.json"

    path = Path(source_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cutoffs: Dict[str, float] = json.load(f)
            return cutoffs

    # Fallback to defaults
    return {
        "general_toxicity": 0.50,
        "severe_abuse": 0.35,
        "obscene_language": 0.45,
        "threatening_language": 0.30,
        "personal_insult": 0.45,
        "identity_attack": 0.35,
    }
