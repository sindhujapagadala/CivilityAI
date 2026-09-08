"""
CivilityAI: Dataset Ingestion & Multi-Label Pipeline.

Handles raw Jigsaw schema ingestion, domain mapping, class imbalance statistics,
PyTorch ModerationTextDataset creation, and stratified batch generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from safety_ml.settings import (
    CATEGORY_ORDER,
    DATASETS_DIR,
    RAW_TO_INTERNAL_COLUMN_MAP,
)
from safety_ml.text_processing import prepare_message_text

logger = logging.getLogger("CivilityAI.DataPipeline")


# Canonical raw column list expected in source Jigsaw CSV
REQUIRED_RAW_COLUMNS = [
    "id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def load_raw_dataset(
    source_file_path: Optional[Path | str] = None,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """
    Loads the source Jigsaw dataset CSV without altering raw column names on initial read.
    Validates the dataset schema and immediately applies domain translation.

    Returns:
        pd.DataFrame containing sanitized 'message_body' and mapped category target columns.
    """
    if source_file_path is None:
        source_file_path = DATASETS_DIR / "source" / "train.csv"

    source_path = Path(source_file_path)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{source_path.resolve()}'. "
            f"Place the Jigsaw 'train.csv' there or run 'python datasets/sample_generator.py'."
        )

    logger.info(f"Loading raw dataset from {source_path}...")
    message_frame = pd.read_csv(source_path)

    # Validate raw schema
    for column_name in REQUIRED_RAW_COLUMNS:
        if column_name not in message_frame.columns:
            raise ValueError(
                f"Missing required raw column '{column_name}' in dataset schema. "
                f"Found columns: {list(message_frame.columns)}"
            )

    if drop_missing:
        initial_count = len(message_frame)
        message_frame = message_frame.dropna(subset=["comment_text"])
        dropped_count = initial_count - len(message_frame)
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows with null comment_text.")

    # Apply internal domain terminology mapping
    # comment_text -> message_body
    message_frame["message_body"] = message_frame["comment_text"].astype(str).apply(prepare_message_text)

    # Map target columns to internal names
    for raw_column, internal_name in RAW_TO_INTERNAL_COLUMN_MAP.items():
        message_frame[internal_name] = message_frame[raw_column].astype(np.float32)

    return message_frame


def compute_class_imbalance_weights(
    message_frame: pd.DataFrame,
    category_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes sample counts, positive prevalence ratios, and positive class weights (pos_weight)
    for torch.nn.BCEWithLogitsLoss to mitigate severe label sparsity.
    """
    if category_columns is None:
        category_columns = CATEGORY_ORDER

    total_samples = len(message_frame)
    category_sample_counts: Dict[str, int] = {}
    positive_class_weights: Dict[str, float] = {}
    prevalence_percentages: Dict[str, float] = {}

    for category in category_columns:
        positive_count = int(message_frame[category].sum())
        negative_count = total_samples - positive_count
        category_sample_counts[category] = positive_count
        prevalence_percentages[category] = (positive_count / max(total_samples, 1)) * 100.0

        # pos_weight = negative_count / (positive_count + eps)
        calculated_weight = negative_count / max(positive_count, 1)
        # Cap the weight to prevent extreme gradient shocks
        positive_class_weights[category] = float(min(calculated_weight, 50.0))

    weight_tensor = torch.tensor(
        [positive_class_weights[cat] for cat in category_columns],
        dtype=torch.float32,
    )

    imbalance_statistics = {
        "total_samples": total_samples,
        "category_sample_counts": category_sample_counts,
        "prevalence_percentages": prevalence_percentages,
        "positive_class_weights": positive_class_weights,
        "weight_tensor": weight_tensor,
    }

    return imbalance_statistics


def partition_dataset(
    message_frame: pd.DataFrame,
    validation_ratio: float = 0.15,
    evaluation_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the message frame into Training, Validation, and Evaluation partitions.
    """
    test_and_val_ratio = validation_ratio + evaluation_ratio
    training_frame, holdout_frame = train_test_split(
        message_frame,
        test_size=test_and_val_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    relative_val_ratio = validation_ratio / test_and_val_ratio
    validation_frame, evaluation_frame = train_test_split(
        holdout_frame,
        test_size=(1.0 - relative_val_ratio),
        random_state=random_seed,
        shuffle=True,
    )

    logger.info(
        f"Partitioned dataset: {len(training_frame)} train, "
        f"{len(validation_frame)} val, {len(evaluation_frame)} test."
    )

    return training_frame, validation_frame, evaluation_frame


class ModerationTextDataset(Dataset):
    """
    PyTorch Dataset encoding text messages into tokenized tensors
    and multi-label safety target vectors.
    """

    def __init__(
        self,
        message_bodies: List[str],
        safety_targets: Optional[np.ndarray | torch.Tensor] = None,
        text_encoder: Any = None,
        max_sequence_length: int = 128,
    ):
        self.message_bodies = list(message_bodies)
        self.max_sequence_length = max_sequence_length
        self.text_encoder = text_encoder

        if safety_targets is not None:
            if isinstance(safety_targets, np.ndarray):
                self.safety_targets = torch.tensor(safety_targets, dtype=torch.float32)
            else:
                self.safety_targets = safety_targets.clone().detach().to(dtype=torch.float32)
        else:
            self.safety_targets = None

    def __len__(self) -> int:
        return len(self.message_bodies)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        message_body = self.message_bodies[index]

        if self.text_encoder is not None:
            encoding = self.text_encoder(
                message_body,
                truncation=True,
                max_length=self.max_sequence_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {
                "encoded_tokens": encoding["input_ids"].squeeze(0),
                "token_visibility": encoding["attention_mask"].squeeze(0),
            }
        else:
            item = {"raw_text": message_body}

        if self.safety_targets is not None:
            item["safety_targets"] = self.safety_targets[index]

        return item


def build_dataloader_partitions(
    training_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    text_encoder: Any,
    max_sequence_length: int = 128,
    training_batch_size: int = 16,
    validation_batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Assembles DataLoader iterators for training, validation, and evaluation.
    """
    category_columns = CATEGORY_ORDER

    training_dataset = ModerationTextDataset(
        message_bodies=training_frame["message_body"].tolist(),
        safety_targets=training_frame[category_columns].values,
        text_encoder=text_encoder,
        max_sequence_length=max_sequence_length,
    )

    validation_dataset = ModerationTextDataset(
        message_bodies=validation_frame["message_body"].tolist(),
        safety_targets=validation_frame[category_columns].values,
        text_encoder=text_encoder,
        max_sequence_length=max_sequence_length,
    )

    evaluation_dataset = ModerationTextDataset(
        message_bodies=evaluation_frame["message_body"].tolist(),
        safety_targets=evaluation_frame[category_columns].values,
        text_encoder=text_encoder,
        max_sequence_length=max_sequence_length,
    )

    training_batches = DataLoader(
        training_dataset,
        batch_size=training_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    validation_batches = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    evaluation_batches = DataLoader(
        evaluation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return training_batches, validation_batches, evaluation_batches
