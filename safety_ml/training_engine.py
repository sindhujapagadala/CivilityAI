"""
CivilityAI: Model Training & Validation Orchestrator.

Orchestrates multi-label training of ContentSafetyTransformer using BCEWithLogitsLoss
with class imbalance weighting, AdamW optimization, linear learning rate scheduling,
validation tracking, and early stopping.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from safety_ml.data_pipeline import (
    build_dataloader_partitions,
    compute_class_imbalance_weights,
    load_raw_dataset,
    partition_dataset,
)
from safety_ml.model_assessment import assess_safety_model, format_assessment_table
from safety_ml.settings import (
    CATEGORY_ORDER,
    SAVED_MODELS_DIR,
    SafetyModelConfig,
)
from safety_ml.transformer_classifier import ContentSafetyTransformer

logger = logging.getLogger("CivilityAI.TrainingEngine")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def select_execution_device(device_preference: str = "auto") -> torch.device:
    """
    Selects CUDA GPU if available and requested, otherwise falls back gracefully to CPU.
    """
    if device_preference == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_preference == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Execution device designated: {device}")
    return device


class SafetyTrainingOrchestrator:
    """
    Executes training rounds, validation checkpointing, and metric evaluations.
    """

    def __init__(
        self,
        config: Optional[SafetyModelConfig] = None,
        output_directory: Optional[Path | str] = None,
    ):
        self.config = config or SafetyModelConfig()
        self.output_directory = Path(output_directory or (SAVED_MODELS_DIR / "transformer"))
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.device = select_execution_device(self.config.device_preference)

        # Initialize text encoder (tokenizer)
        logger.info(f"Initializing tokenizer for '{self.config.model_name}'...")
        self.text_encoder = AutoTokenizer.from_pretrained(self.config.model_name)

        # Instantiate ContentSafetyTransformer
        logger.info("Instantiating ContentSafetyTransformer architecture...")
        self.language_guard = ContentSafetyTransformer(
            base_model_identifier=self.config.model_name,
            num_safety_categories=len(CATEGORY_ORDER),
            dropout_probability=self.config.dropout_probability,
        ).to(self.device)

    def train_epoch(
        self,
        training_batches: Any,
        parameter_optimizer: torch.optim.Optimizer,
        learning_rate_scheduler: Any,
        loss_criterion: nn.BCEWithLogitsLoss,
        max_gradient_norm: float = 1.0,
    ) -> float:
        """
        Runs one complete pass through all training batches.
        """
        self.language_guard.train()
        accumulated_loss = 0.0
        total_batches = len(training_batches)

        for batch_index, sample_batch in enumerate(training_batches, 1):
            encoded_tokens = sample_batch["encoded_tokens"].to(self.device)
            token_visibility = sample_batch["token_visibility"].to(self.device)
            safety_targets = sample_batch["safety_targets"].to(self.device)

            parameter_optimizer.zero_grad()

            category_logits = self.language_guard(encoded_tokens, token_visibility)
            training_loss = loss_criterion(category_logits, safety_targets)

            training_loss.backward()
            nn.utils.clip_grad_norm_(self.language_guard.parameters(), max_gradient_norm)

            parameter_optimizer.step()
            learning_rate_scheduler.step()

            accumulated_loss += training_loss.item()

            if batch_index % 10 == 0 or batch_index == total_batches:
                logger.info(
                    f"Batch {batch_index}/{total_batches} - Current Loss: {training_loss.item():.4f}"
                )

        average_loss = accumulated_loss / max(total_batches, 1)
        return average_loss

    def evaluate_model(
        self,
        evaluation_batches: Any,
        loss_criterion: nn.BCEWithLogitsLoss,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Runs inference over holdout batches without gradient calculation.
        """
        self.language_guard.eval()
        accumulated_loss = 0.0
        all_targets: list[np.ndarray] = []
        all_probabilities: list[np.ndarray] = []

        with torch.no_grad():
            for sample_batch in evaluation_batches:
                encoded_tokens = sample_batch["encoded_tokens"].to(self.device)
                token_visibility = sample_batch["token_visibility"].to(self.device)
                safety_targets = sample_batch["safety_targets"].to(self.device)

                category_logits = self.language_guard(encoded_tokens, token_visibility)
                eval_loss = loss_criterion(category_logits, safety_targets)

                accumulated_loss += eval_loss.item()

                risk_probabilities = torch.sigmoid(category_logits)
                all_targets.append(safety_targets.cpu().numpy())
                all_probabilities.append(risk_probabilities.cpu().numpy())

        mean_loss = accumulated_loss / max(len(evaluation_batches), 1)
        concatenated_targets = np.vstack(all_targets)
        concatenated_probs = np.vstack(all_probabilities)

        return mean_loss, concatenated_targets, concatenated_probs

    def run_complete_training_cycle(
        self,
        dataset_csv_path: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """
        Executes the entire training pipeline: data preparation, class weighting,
        multi-round training, early stopping, and metric persistence.
        """
        # 1. Dataset Loading & Partitioning
        message_frame = load_raw_dataset(dataset_csv_path)
        imbalance_stats = compute_class_imbalance_weights(message_frame)
        positive_class_weights = imbalance_stats["weight_tensor"].to(self.device)

        logger.info(f"Configured positive class weights for BCE loss: {positive_class_weights.tolist()}")

        training_frame, validation_frame, evaluation_frame = partition_dataset(
            message_frame,
            random_seed=self.config.random_seed,
        )

        training_batches, validation_batches, evaluation_batches = build_dataloader_partitions(
            training_frame=training_frame,
            validation_frame=validation_frame,
            evaluation_frame=evaluation_frame,
            text_encoder=self.text_encoder,
            max_sequence_length=self.config.max_sequence_length,
            training_batch_size=self.config.training_batch_size,
            validation_batch_size=self.config.validation_batch_size,
        )

        # 2. Multi-Label Loss Formulation with pos_weight
        # BCEWithLogitsLoss combines Sigmoid and Binary Cross Entropy with numerical stability
        loss_criterion = nn.BCEWithLogitsLoss(pos_weight=positive_class_weights)

        # 3. Parameter Optimizer & Warmup Scheduler
        parameter_optimizer = AdamW(
            self.language_guard.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_training_steps = len(training_batches) * self.config.training_rounds
        warmup_steps = int(total_training_steps * self.config.warmup_ratio)

        learning_rate_scheduler = get_linear_schedule_with_warmup(
            parameter_optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        # 4. Training Loop with Early Stopping
        best_validation_loss = float("inf")
        patience_counter = 0
        best_assessment_report: Dict[str, Any] = {}

        logger.info(f"Starting {self.config.training_rounds} training rounds...")

        for round_number in range(1, self.config.training_rounds + 1):
            logger.info(f"--- Training Round {round_number}/{self.config.training_rounds} ---")
            train_loss = self.train_epoch(
                training_batches=training_batches,
                parameter_optimizer=parameter_optimizer,
                learning_rate_scheduler=learning_rate_scheduler,
                loss_criterion=loss_criterion,
            )

            val_loss, val_targets, val_probs = self.evaluate_model(
                evaluation_batches=validation_batches,
                loss_criterion=loss_criterion,
            )

            round_assessment = assess_safety_model(
                true_targets=val_targets,
                risk_probabilities=val_probs,
                category_columns=CATEGORY_ORDER,
            )

            logger.info(
                f"Round {round_number} Complete - "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Macro-F1: {round_assessment['macro_f1']:.4f} | "
                f"Val Micro-F1: {round_assessment['micro_f1']:.4f}"
            )

            # Checkpoint saving on best validation loss
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_assessment_report = round_assessment
                patience_counter = 0
                self.save_model_artifacts(round_assessment)
                logger.info(f"New best model checkpoint saved at round {round_number}.")
            else:
                patience_counter += 1
                logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{self.config.early_stopping_patience}")
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info("Early stopping triggered.")
                    break

        # 5. Final Evaluation on Holdout Evaluation Set
        logger.info("Executing final assessment on Holdout Evaluation partition...")
        eval_loss, eval_targets, eval_probs = self.evaluate_model(
            evaluation_batches=evaluation_batches,
            loss_criterion=loss_criterion,
        )
        final_evaluation_report = assess_safety_model(
            true_targets=eval_targets,
            risk_probabilities=eval_probs,
            category_columns=CATEGORY_ORDER,
        )

        logger.info("\n" + "=" * 90)
        logger.info("DISTILBERT TRANSFORMER: FINAL HOLDOUT EVALUATION REPORT")
        logger.info("=" * 90)
        logger.info("\n" + format_assessment_table(final_evaluation_report))
        logger.info("=" * 90)

        # Save metrics to json
        metrics_file = self.output_directory / "evaluation_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(final_evaluation_report, f, indent=2)

        return final_evaluation_report

    def save_model_artifacts(self, assessment_report: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves the tokenizer, transformer state dictionary, and training metadata.
        """
        weights_path = self.output_directory / "safety_guard_weights.pt"
        torch.save(self.language_guard.state_dict(), weights_path)
        self.text_encoder.save_pretrained(self.output_directory)

        metadata = {
            "model_name": self.config.model_name,
            "max_sequence_length": self.config.max_sequence_length,
            "categories": CATEGORY_ORDER,
            "assessment_report": assessment_report,
        }
        with open(self.output_directory / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved transformer model artifacts to {self.output_directory.resolve()}")


if __name__ == "__main__":
    orchestrator = SafetyTrainingOrchestrator()
    orchestrator.run_complete_training_cycle()
