"""
CivilityAI: Exploratory Data Analysis (EDA) Engine.

Analyzes dataset characteristics, class distributions, label co-occurrences,
and text length properties, outputting figures to analysis_outputs/figures/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from safety_ml.data_pipeline import load_raw_dataset
from safety_ml.settings import (
    ANALYSIS_OUTPUTS_DIR,
    CATEGORY_ORDER,
    DATASETS_DIR,
)

logger = logging.getLogger("CivilityAI.EDA")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def execute_exploratory_analysis(
    source_file_path: Path | str | None = None,
    output_figures_dir: Path | str | None = None,
) -> Dict[str, Any]:
    """
    Runs full exploratory data analysis across the toxicity dataset.
    """
    if source_file_path is None:
        source_file_path = DATASETS_DIR / "source" / "train.csv"
    if output_figures_dir is None:
        output_figures_dir = ANALYSIS_OUTPUTS_DIR / "figures"

    figures_dir = Path(output_figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion
    message_frame = load_raw_dataset(source_file_path, drop_missing=False)
    total_message_count = len(message_frame)
    category_columns = CATEGORY_ORDER

    # 2. Missing & Duplicate Analysis
    missing_entries = int(message_frame["comment_text"].isna().sum())
    duplicate_messages = int(message_frame.duplicated(subset=["comment_text"]).sum())

    # 3. Message Length Statistics
    message_lengths = message_frame["message_body"].str.len()
    average_message_length = float(message_lengths.mean())
    maximum_message_length = int(message_lengths.max())
    minimum_message_length = int(message_lengths.min())
    median_message_length = float(message_lengths.median())

    # 4. Safe vs Unsafe Distribution
    # A message is toxic/unsafe if any of the 6 category targets is 1
    any_category_active = message_frame[category_columns].sum(axis=1) > 0
    unsafe_message_count = int(any_category_active.sum())
    safe_message_count = total_message_count - unsafe_message_count
    safe_percentage = (safe_message_count / max(total_message_count, 1)) * 100.0
    unsafe_percentage = (unsafe_message_count / max(total_message_count, 1)) * 100.0

    # 5. Samples per category & Imbalance
    category_frequency: Dict[str, int] = {}
    category_percentages: Dict[str, float] = {}
    for cat in category_columns:
        freq = int(message_frame[cat].sum())
        category_frequency[cat] = freq
        category_percentages[cat] = (freq / max(total_message_count, 1)) * 100.0

    # 6. Multi-label combinations
    active_labels_per_message = message_frame[category_columns].sum(axis=1)
    label_count_distribution = active_labels_per_message.value_counts().sort_index().to_dict()

    # Visualizations styling
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
    palette = sns.color_palette("mako", len(category_columns))

    # Figure 1: Safe vs Unsafe Overview Pie / Bar Chart
    plt.figure(figsize=(8, 5))
    bar_colors = ["#2ecc71", "#e74c3c"]
    bars = plt.bar(["Safe Content", "Unsafe / Flagged"], [safe_message_count, unsafe_message_count], color=bar_colors, width=0.5)
    plt.title("CivilityAI: Safe vs. Unsafe Content Distribution", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Number of Messages", fontsize=11)
    for bar in bars:
        height = bar.get_height()
        pct = (height / max(total_message_count, 1)) * 100
        plt.annotate(
            f"{int(height):,} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.tight_layout()
    fig1_path = figures_dir / "safe_vs_toxic_overview.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Figure 2: Category Frequency & Severe Class Imbalance
    plt.figure(figsize=(10, 5.5))
    categories_clean_names = [cat.replace("_", " ").title() for cat in category_columns]
    freq_values = [category_frequency[cat] for cat in category_columns]
    bars = plt.barh(categories_clean_names, freq_values, color=palette)
    plt.title("Prevalence across Safety Categories (Class Imbalance)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Positive Sample Count", fontsize=11)
    for bar in bars:
        width = bar.get_width()
        pct = (width / max(total_message_count, 1)) * 100
        plt.annotate(
            f"{int(width):,} ({pct:.1f}%)",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=9,
        )
    plt.tight_layout()
    fig2_path = figures_dir / "class_imbalance_distribution.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # Figure 3: Message Length Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(message_lengths, bins=40, color="#3498db", kde=True)
    plt.title("Message Length Distribution (Characters)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Message Length (Characters)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.axvline(average_message_length, color="#e74c3c", linestyle="--", label=f"Mean: {average_message_length:.1f}")
    plt.axvline(median_message_length, color="#f39c12", linestyle=":", label=f"Median: {median_message_length:.1f}")
    plt.legend(frameon=True)
    plt.tight_layout()
    fig3_path = figures_dir / "message_length_distribution.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # Figure 4: Multi-Label Correlation Matrix
    plt.figure(figsize=(8, 6.5))
    correlation_matrix = message_frame[category_columns].corr()
    clean_labels = [c.replace("_", "\n").title() for c in category_columns]
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        cbar_kws={"label": "Pearson Correlation"},
    )
    plt.title("Multi-Label Category Co-occurrence Correlation", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    fig4_path = figures_dir / "multilabel_cooccurrence_matrix.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    summary_results = {
        "total_message_count": total_message_count,
        "safe_message_count": safe_message_count,
        "unsafe_message_count": unsafe_message_count,
        "safe_percentage": safe_percentage,
        "unsafe_percentage": unsafe_percentage,
        "category_frequency": category_frequency,
        "category_percentages": category_percentages,
        "average_message_length": average_message_length,
        "maximum_message_length": maximum_message_length,
        "minimum_message_length": minimum_message_length,
        "median_message_length": median_message_length,
        "missing_entries": missing_entries,
        "duplicate_messages": duplicate_messages,
        "label_count_distribution": label_count_distribution,
        "saved_figures": [str(fig1_path), str(fig2_path), str(fig3_path), str(fig4_path)],
    }

    # Save summary report JSON
    report_path = figures_dir.parent / "reports" / "eda_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as json_out:
        json.dump(summary_results, json_out, indent=2)

    logger.info(f"EDA successfully finished. Figures saved in {figures_dir.resolve()}")
    logger.info(f"Summary JSON saved in {report_path.resolve()}")
    return summary_results


if __name__ == "__main__":
    execute_exploratory_analysis()
