"""
CivilityAI: Model Error Inspection & Vulnerability Analysis.

Diagnoses model blindspots, identifying false positives, false negatives,
high-confidence mistakes, and uncertain predictions across safety categories.
Exports diagnostic summaries to CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from safety_ml.settings import ANALYSIS_OUTPUTS_DIR, CATEGORY_ORDER

logger = logging.getLogger("CivilityAI.ErrorInspection")


def inspect_model_errors(
    message_bodies: List[str],
    ground_truth_targets: np.ndarray,
    predicted_probabilities: np.ndarray,
    decision_cutoffs: Dict[str, float],
    category_columns: Optional[List[str]] = None,
    output_directory: Optional[Path | str] = None,
    uncertainty_band: float = 0.10,
) -> Dict[str, Any]:
    """
    Performs deep diagnosis of model mistakes and confidence boundaries.

    Args:
        message_bodies: List of message strings evaluated.
        ground_truth_targets: Binary ground truth array (N, num_categories).
        predicted_probabilities: Model probabilities (N, num_categories).
        decision_cutoffs: Dictionary mapping category to threshold.
        category_columns: Names of safety categories.
        output_directory: Directory where diagnostic CSVs are exported.
        uncertainty_band: Margin around threshold where predictions are deemed uncertain.

    Returns:
        Dictionary containing counts, summary tables, and case records.
    """
    if category_columns is None:
        category_columns = CATEGORY_ORDER

    reports_dir = Path(output_directory or (ANALYSIS_OUTPUTS_DIR / "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    cutoff_vector = np.array([decision_cutoffs.get(cat, 0.50) for cat in category_columns])
    binary_predictions = (predicted_probabilities >= cutoff_vector).astype(int)

    false_positive_cases: List[Dict[str, Any]] = []
    false_negative_cases: List[Dict[str, Any]] = []
    high_confidence_mistakes: List[Dict[str, Any]] = []
    uncertain_messages: List[Dict[str, Any]] = []
    category_error_summary: Dict[str, Dict[str, int]] = {}

    for col_idx, category in enumerate(category_columns):
        cutoff = cutoff_vector[col_idx]
        category_fp = 0
        category_fn = 0

        for row_idx, message in enumerate(message_bodies):
            truth = int(ground_truth_targets[row_idx, col_idx])
            prob = float(predicted_probabilities[row_idx, col_idx])
            pred = int(binary_predictions[row_idx, col_idx])

            # False Positive: predicted 1, truth is 0
            if pred == 1 and truth == 0:
                category_fp += 1
                record = {
                    "message_body": message,
                    "category": category,
                    "ground_truth": 0,
                    "predicted_probability": round(prob, 4),
                    "decision_cutoff": round(cutoff, 3),
                    "error_type": "False Positive",
                }
                false_positive_cases.append(record)

                if prob >= 0.85:
                    high_confidence_mistakes.append({**record, "severity": "High-Confidence False Alarm"})

            # False Negative: predicted 0, truth is 1
            elif pred == 0 and truth == 1:
                category_fn += 1
                record = {
                    "message_body": message,
                    "category": category,
                    "ground_truth": 1,
                    "predicted_probability": round(prob, 4),
                    "decision_cutoff": round(cutoff, 3),
                    "error_type": "False Negative",
                }
                false_negative_cases.append(record)

                if prob <= 0.15:
                    high_confidence_mistakes.append({**record, "severity": "High-Confidence False Neglect"})

            # Uncertainty boundary check: probability within [cutoff - band, cutoff + band]
            if abs(prob - cutoff) <= uncertainty_band:
                uncertain_messages.append({
                    "message_body": message,
                    "category": category,
                    "ground_truth": truth,
                    "predicted_probability": round(prob, 4),
                    "distance_to_cutoff": round(abs(prob - cutoff), 4),
                })

        category_error_summary[category] = {
            "false_positives": category_fp,
            "false_negatives": category_fn,
            "total_errors": category_fp + category_fn,
        }

    # Export to CSV files for auditing
    fp_df = pd.DataFrame(false_positive_cases)
    fn_df = pd.DataFrame(false_negative_cases)
    hc_df = pd.DataFrame(high_confidence_mistakes)
    unc_df = pd.DataFrame(uncertain_messages)

    fp_csv_path = reports_dir / "diagnostic_false_positives.csv"
    fn_csv_path = reports_dir / "diagnostic_false_negatives.csv"
    hc_csv_path = reports_dir / "diagnostic_high_confidence_mistakes.csv"
    unc_csv_path = reports_dir / "diagnostic_uncertain_predictions.csv"

    if not fp_df.empty:
        fp_df.to_csv(fp_csv_path, index=False)
    if not fn_df.empty:
        fn_df.to_csv(fn_csv_path, index=False)
    if not hc_df.empty:
        hc_df.to_csv(hc_csv_path, index=False)
    if not unc_df.empty:
        unc_df.to_csv(unc_csv_path, index=False)

    logger.info(f"Error diagnostics written to {reports_dir.resolve()}")

    return {
        "category_error_summary": category_error_summary,
        "total_false_positives": len(false_positive_cases),
        "total_false_negatives": len(false_negative_cases),
        "total_high_confidence_mistakes": len(high_confidence_mistakes),
        "total_uncertain_messages": len(uncertain_messages),
        "exported_files": [
            str(fp_csv_path),
            str(fn_csv_path),
            str(hc_csv_path),
            str(unc_csv_path),
        ],
    }
