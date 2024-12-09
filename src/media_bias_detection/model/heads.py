"""Model heads implementation for MTL model.

This module contains all task-specific heads and the factory for creating them.
Each head implements specific logic for different types of tasks while maintaining
consistent interfaces for the MTL architecture.
"""

from typing import Dict, Tuple, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, MeanSquaredError, Perplexity, R2Score

from .gradient import GradsWrapper
from ..tokenizer import tokenizer
from ..utils.logger import general_logger
from ..utils.common import get_class_weights
from ..data.task import (
    SubTask,
    ClassificationSubTask,
    MultiLabelClassificationSubTask,
    POSSubTask,
    RegressionSubTask,
    MLMSubTask
)


class HeadError(Exception):
    """Custom exception for head-related errors."""
    pass


def HeadFactory(st: SubTask, *args, **kwargs) -> 'BaseHead':
    """Create appropriate head based on subtask type.

    Args:
        st: Subtask to create head for
        *args, **kwargs: Additional arguments for head initialization

    Returns:
        Initialized head instance

    Raises:
        HeadError: If head creation fails or subtask type is unsupported
    """
    try:
        if isinstance(st, ClassificationSubTask):
            print(f"Creating ClassificationHead for subtask {st.id}")
            print(f"num_classes: {st.num_classes}")
            return ClassificationHead(
                num_classes=st.num_classes,
                class_weights=st.class_weights,
                *args,
                **kwargs
            )
        elif isinstance(st, MultiLabelClassificationSubTask):
            print(f"Creating MultiLabelClassificationHead for subtask {st.id}")
            print(f"num_classes: {st.num_classes}, num_labels: {st.num_labels}")
            return ClassificationHead(
                num_classes=st.num_classes,
                num_labels=st.num_labels if st.num_labels is not None else 2,
                class_weights=st.class_weights,
                *args,
                **kwargs
            )
        elif isinstance(st, POSSubTask):
            return TokenClassificationHead(
                num_classes=st.num_classes,
                class_weights=st.class_weights,
                *args,
                **kwargs
            )
        elif isinstance(st, RegressionSubTask):
            return RegressionHead(*args, **kwargs)
        elif isinstance(st, MLMSubTask):
            return LanguageModellingHead(*args, **kwargs)
        else:
            raise HeadError(f"Unsupported subtask type: {type(st)}")
    except Exception as e:
        raise HeadError(f"Head creation failed: {str(e)}")


class BaseHead(GradsWrapper):
    """Base class for all model heads.

    Attributes:
        metrics: Dictionary of metric names to metric instances
    """

    def __init__(self):
        super().__init__()
        self.metrics: Dict = {}

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Forward pass through the head.

        Args:
            X: Input features (batch_size, seq_len, hidden_dim)
            y: Target labels

        Returns:
            Tuple of (logits, loss, metric_values)
        """
        raise NotImplementedError


class ClassificationHead(BaseHead):
    def __init__(
            self,
            input_dimension: int,
            hidden_dimension: int,
            dropout_prob: float,
            num_classes: int = 2,
            num_labels: int = 1,
            class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        print(f"Initializing ClassificationHead")
        print(f"num_classes: {num_classes}")
        print(f"num_labels: {num_labels}")

        # Common layers
        self.dense = nn.Linear(input_dimension, hidden_dimension)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.out_proj = nn.Linear(hidden_dimension, num_classes * num_labels)

        # Store dimensions
        self.num_classes = num_classes
        self.num_labels = num_labels

        # Use CrossEntropyLoss for both cases
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Initializing ClassificationHead with {num_labels} labels")
        # Set up metrics based on task type
        if num_labels > 1:  # Multi-label case
            self.metrics = {
                "f1": F1Score(
                    num_classes=num_classes,
                    num_labels=num_labels,
                    task="multilabel",
                    average="macro"
                ),
                "acc": Accuracy(
                    task="multilabel",
                    num_classes=num_classes,
                    num_labels=num_labels,
                ),
            }
        else:  # Regular classification case
            self.metrics = {
                "f1": F1Score(
                    num_classes=num_classes,
                    task="binary" if num_classes == 2 else "multiclass",
                    average="macro"
                ),
                "acc": Accuracy(
                    task="binary" if num_classes == 2 else "multiclass",
                    num_classes=num_classes,
                ),
            }

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        try:
            batch_size = y.shape[0]
            print(f"Batch size: {batch_size}")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")

            # Get CLS token representation
            x = X[:, 0, :]  # take <s> token (equiv. to [CLS])

            # Pass through layers
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            logits = self.out_proj(x)

            # Compute loss
            loss = self.loss(logits.view(-1, self.num_classes), y.view(-1))
            print(f"Logits shape: {logits.shape}")

            # Reshape logits based on task type
            if self.num_labels > 1:  # Multi-label case
                logits = logits.view(batch_size, self.num_labels, self.num_classes)
                y = y.view(batch_size, self.num_labels)
            else:  # Binary/multiclass case
                logits = logits.view(batch_size, self.num_classes)
                y = y.view(batch_size)  # Flatten targets

            # Compute loss
            loss = self.loss(logits, y)

            # Get predictions in correct shape for metrics
            predictions = torch.argmax(logits, dim=-1)  # Use last dimension for class prediction

            # Calculate metrics
            metrics = {
                name: metric(predictions.cpu(), y.cpu())
                for name, metric in self.metrics.items()
            }

            return logits, loss, metrics

        except Exception as e:
            raise HeadError(f"Classification forward pass failed: {str(e)}")


class TokenClassificationHead(BaseHead):
    """Head for token-level classification tasks.

    Attributes:
        dropout: Dropout layer
        classifier: Classification layer
        num_classes: Number of classes
        loss: Loss function
        metrics: Dictionary of metrics
    """

    def __init__(
            self,
            num_classes: int,
            class_weights: Optional[torch.Tensor],
            hidden_dimension: int,
            dropout_prob: float,
            *args,
            **kwargs
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_dimension, num_classes)
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

        self.metrics = {
            "f1": F1Score(
                num_classes=num_classes,
                task="multiclass",
                average="macro"
            ),
            "acc": Accuracy(
                task="multiclass",
                num_classes=num_classes
            ),
        }

        general_logger.info(
            f"Initialized TokenClassificationHead with {num_classes} classes"
        )

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        try:
            # Process sequence
            sequence_output = self.dropout(X)
            logits = self.classifier(sequence_output)

            # Compute loss
            loss = self.loss(logits.view(-1, self.num_classes), y.view(-1))

            # Mask padding tokens for metrics
            mask = torch.where(y != -100, 1, 0)
            logits = torch.masked_select(
                logits,
                mask.unsqueeze(-1).expand(logits.size()) == 1
            )
            y = torch.masked_select(y, mask == 1)
            logits = logits.view(y.shape[0], self.num_classes)

            # calculate metrics with predictions instead of logits
            predictions = torch.argmax(logits, dim=1)
            metrics = {
                name: metric(predictions.cpu(), y.cpu())
                for name, metric in self.metrics.items()
            }

            return logits, loss, metrics

        except Exception as e:
            raise HeadError(f"Token classification forward pass failed: {str(e)}")


class RegressionHead(BaseHead):
    """Head for regression tasks.

    Attributes:
        dense: Dense layer
        dropout: Dropout layer
        out_proj: Output projection layer
        loss: Loss function
        metrics: Dictionary of metrics
    """

    def __init__(
            self,
            input_dimension: int,
            hidden_dimension: int,
            dropout_prob: float
    ):
        super().__init__()

        self.dense = nn.Linear(input_dimension, hidden_dimension)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.out_proj = nn.Linear(hidden_dimension, 1)

        self.loss = nn.MSELoss()
        self.metrics = {
            "R2": R2Score(),
            "MSE": MeanSquaredError()
        }

        general_logger.info("Initialized RegressionHead")

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        try:
            # Get CLS token
            x = X[:, 0, :]

            # Pass through layers
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            logits = self.out_proj(x)

            loss = self.loss(logits.squeeze(), y.squeeze())

            metrics = {
                name: metric(logits.cpu(), y.cpu()).detach()
                for name, metric in self.metrics.items()
            }

            return logits, loss, metrics

        except Exception as e:
            raise HeadError(f"Regression forward pass failed: {str(e)}")


class LanguageModellingHead(BaseHead):
    """Head for masked language modeling tasks.

    Attributes:
        dense: Dense layer
        layer_norm: Layer normalization
        decoder: Output decoder
        loss: Loss function
        metrics: Dictionary of metrics
    """

    def __init__(
            self,
            input_dimension: int,
            hidden_dimension: int,
            dropout_prob: float
    ):
        super().__init__()

        self.dense = nn.Linear(input_dimension, hidden_dimension)
        self.layer_norm = nn.LayerNorm(hidden_dimension, eps=1e-5)
        self.gelu = nn.GELU()

        self.decoder = nn.Linear(hidden_dimension, tokenizer.vocab_size)
        self.bias = nn.Parameter(torch.zeros(tokenizer.vocab_size))
        self.decoder.bias = self.bias

        self.loss = nn.CrossEntropyLoss()
        self.metrics = {"perplexity": Perplexity()}

        general_logger.info("Initialized LanguageModellingHead")

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        try:
            x = self.dense(X)
            x = self.gelu(x)
            x = self.layer_norm(x)

            logits = self.decoder(x)
            loss = self.loss(
                logits.view(-1, tokenizer.vocab_size),
                y.view(-1)
            )

            metrics = {
                name: metric(logits.cpu(), y.cpu())
                for name, metric in self.metrics.items()
            }

            return logits, loss, metrics

        except Exception as e:
            raise HeadError(f"Language modeling forward pass failed: {str(e)}")