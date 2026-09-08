"""
CivilityAI: Transformer-Based Content Safety Classifier.

Implements ContentSafetyTransformer using DistilBERT encoder and a dedicated
multi-label safety classification head for 6 toxicity categories.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

logger = logging.getLogger("CivilityAI.TransformerClassifier")


class ContentSafetyTransformer(nn.Module):
    """
    Production Deep Learning Architecture for Multi-Label Content Moderation.

    Passes encoded tokens through DistilBERT to extract contextual language representations,
    then projects pooled sequence features through a multi-category safety classification head.
    """

    def __init__(
        self,
        base_model_identifier: str = "distilbert-base-uncased",
        num_safety_categories: int = 6,
        dropout_probability: float = 0.20,
    ):
        super().__init__()
        self.num_safety_categories = num_safety_categories
        self.base_model_identifier = base_model_identifier

        # Load contextual language encoder configuration
        self.encoder_config = AutoConfig.from_pretrained(
            base_model_identifier,
            output_hidden_states=False,
        )
        self.language_encoder = AutoModel.from_pretrained(
            base_model_identifier,
            config=self.encoder_config,
        )

        hidden_dimension = self.encoder_config.hidden_size  # Typically 768 for DistilBERT

        # Multi-layer safety classification head
        self.safety_classification_head = nn.Sequential(
            nn.Dropout(dropout_probability),
            nn.Linear(hidden_dimension, 256),
            nn.GELU(),
            nn.Dropout(dropout_probability),
            nn.Linear(256, num_safety_categories),
        )

        # Sigmoid activation converts unbounded logits into independent category probabilities
        self.probability_activation = nn.Sigmoid()

    def forward(
        self,
        encoded_tokens: torch.Tensor,
        token_visibility: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward inference pass.

        Args:
            encoded_tokens: Tensor of token ids (batch_size, sequence_length).
            token_visibility: Attention mask indicating non-padding tokens.

        Returns:
            category_logits: Raw unbounded logits for 6 safety categories (batch_size, 6).
        """
        encoder_output = self.language_encoder(
            input_ids=encoded_tokens,
            attention_mask=token_visibility,
        )

        # DistilBERT contextual representation: First token sequence output corresponds to [CLS]
        # Shape: (batch_size, hidden_dimension)
        context_representation = encoder_output.last_hidden_state[:, 0, :]

        # Compute raw category logits
        category_logits = self.safety_classification_head(context_representation)
        return category_logits

    def compute_risk_probabilities(
        self,
        encoded_tokens: torch.Tensor,
        token_visibility: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method that maps forward logits directly through Sigmoid.
        """
        category_logits = self.forward(encoded_tokens, token_visibility)
        risk_probabilities = self.probability_activation(category_logits)
        return risk_probabilities
