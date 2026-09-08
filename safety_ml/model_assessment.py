"""
CivilityAI: Model Assessment & Multi-Label Metrics Evaluator.

Computes comprehensive evaluation metrics for multi-label safety classification:
Precision, Recall, F1, ROC-AUC, PR-AUC, Micro-F1, Macro-F1, and Weighted-F1.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from safety_ml.settings import CATEGORY_ORDER

logger = logging.getLogger("CivilityAI.ModelAssessment")


def assess_safety_model(
    true_targets: np.ndarray,
    risk_probabilities: np.ndarray,
    decision_cutoffs: Optional[Dict[str, float] | np.ndarray] = None,
    category_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes rigorous evaluation metrics across all 6 safety categories.

    Args:
        true_targets: Binary ground truth array of shape (N, num_categories).
        risk_probabilities: Predicted probabilities in range [0, 1] of shape (N, num_categories).
        decision_cutoffs: Cutoffs used to convert probabilities into binary predictions.
                          Defaults to 0.50 for all categories if omitted.
        category_columns: List of category names matching columns of target arrays.

    Returns:
        Structured dictionary containing global aggregations (macro, micro, weighted)
        and per-category detailed metrics.
    """
    if category_columns is None:
        category_columns = CATEGORY_ORDER

    num_categories = len(category_columns)
    if decision_cutoffs is None:
        cutoff_vector = np.full(num_categories, 0.50, dtype=np.float32)
    elif isinstance(decision_cutoffs, dict):
        cutoff_vector = np.array(
            [decision_cutoffs.get(cat, 0.50) for cat in category_columns],
            dtype=np.float32,
        )
    else:
        cutoff_vector = np.asarray(decision_cutoffs, dtype=np.float32)

    # Threshold probabilities to derive binary predictions
    binary_safety_predictions = (risk_probabilities >= cutoff_vector).astype(int)

    # 1. Macro, Micro, Weighted Aggregated Metrics
    macro_precision = float(precision_score(true_targets, binary_safety_predictions, average="macro", zero_division=0))
    macro_recall = float(recall_score(true_targets, binary_safety_predictions, average="macro", zero_division=0))
    macro_f1 = float(f1_score(true_targets, binary_safety_predictions, average="macro", zero_division=0))

    micro_precision = float(precision_score(true_targets, binary_safety_predictions, average="micro", zero_division=0))
    micro_recall = float(recall_score(true_targets, binary_safety_predictions, average="micro", zero_division=0))
    micro_f1 = float(f1_score(true_targets, binary_safety_predictions, average="micro", zero_division=0))

    weighted_f1 = float(f1_score(true_targets, binary_safety_predictions, average="weighted", zero_division=0))

    # ROC-AUC and PR-AUC (Average Precision)
    roc_auc_values: List[float] = []
    pr_auc_values: List[float] = []
    category_assessments: Dict[str, Dict[str, float]] = {}

    for idx, category in enumerate(category_columns):
        cat_true = true_targets[:, idx]
        cat_prob = risk_probabilities[:, idx]
        cat_pred = binary_safety_predictions[:, idx]

        cat_prec = float(precision_score(cat_true, cat_pred, zero_division=0))
        cat_rec = float(recall_score(cat_true, cat_pred, zero_division=0))
        cat_f1 = float(f1_score(cat_true, cat_pred, zero_division=0))

        # Check if category has both classes in ground truth
        has_both_classes = len(np.unique(cat_true)) > 1
        if has_both_classes:
            try:
                cat_roc_auc = float(roc_auc_score(cat_true, cat_prob))
            except Exception:
                cat_roc_auc = 0.50

            try:
                cat_pr_auc = float(average_precision_score(cat_true, cat_prob))
            except Exception:
                cat_pr_auc = 0.0
        else:
            cat_roc_auc = 0.50
            cat_pr_auc = 0.0

        roc_auc_values.append(cat_roc_auc)
        pr_auc_values.append(cat_pr_auc)

        category_assessments[category] = {
            "precision": cat_prec,
            "recall": cat_rec,
            "f1_score": cat_f1,
            "roc_auc": cat_roc_auc,
            "pr_auc": cat_pr_auc,
            "decision_cutoff": float(cutoff_vector[idx]),
            "positive_support": int(np.sum(cat_true)),
        }

    macro_roc_auc = float(np.mean(roc_auc_values))
    macro_pr_auc = float(np.mean(pr_auc_values))

    assessment_report = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "macro_roc_auc": macro_roc_auc,
        "macro_pr_auc": macro_pr_auc,
        "category_metrics": category_assessments,
    }

    return assessment_report


def format_assessment_table(assessment_report: Dict[str, Any]) -> str:
    """
    Renders a clean text table comparing category performance.
    """
    header = (
        f"{'Category':<24} | {'Precision':<9} | {'Recall':<7} | {'F1':<6} | "
        f"{'ROC-AUC':<8} | {'PR-AUC':<7} | {'Cutoff':<6} | {'Support':<7}\n"
        + "-" * 90
    )
    lines = [header]
    for category, metrics in assessment_report["category_metrics"].items():
        clean_name = category.replace("_", " ").title()
        line = (
            f"{clean_name:<24} | {metrics['precision']:<9.4f} | {metrics['recall']:<7.4f} | "
            f"{metrics['f1_score']:<6.4f} | {metrics['roc_auc']:<8.4f} | "
            f"{metrics['pr_auc']:<7.4f} | {metrics['decision_cutoff']:<6.2f} | "
            f"{metrics['positive_support']:<7d}"
        )
        lines.append(line)

    summary = (
        "-" * 90 + "\n"
        f"MACRO AVERAGES: Precision={assessment_report['macro_precision']:.4f} | "
        f"Recall={assessment_report['macro_recall']:.4f} | Macro-F1={assessment_report['macro_f1']:.4f} | "
        f"ROC-AUC={assessment_report['macro_roc_auc']:.4f} | PR-AUC={assessment_report['macro_pr_auc']:.4f}\n"
        f"MICRO-F1: {assessment_report['micro_f1']:.4f} | WEIGHTED-F1: {assessment_report['weighted_f1']:.4f}"
    )
    lines.append(summary)
    return "\n".join(lines)
