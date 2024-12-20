"""Model heads implementation for MTL model.

This module handles the task-specific processing parts of the model:


Types of Heads:
1. ClassificationHead: For yes/no or multiple choice tasks
   - Binary classification (e.g., biased/unbiased)
   - Multi-class classification (e.g., emotion types)

2. TokenClassificationHead: For word-level tasks
   - Identifying specific words (e.g., biased terms)
   - Marking spans of text

Features:
- Factory pattern to create appropriate heads
- Common interface across head types
- Automatic metric tracking (accuracy, F1 score)
- Handles both single and multi-label tasks

Note: Each head takes features from the backbone and processes them
for its specific task. Think of heads as specialized experts that
each look at the same text but focus on different aspects.
"""

from typing import Dict, Tuple, Optional

import torch
from torch import nn
from torchmetrics import Accuracy, F1Score

from media_bias_detection.training.gradient import GradsWrapper
from ..data.task import (
    SubTask,
    ClassificationSubTask,
    MultiLabelClassificationSubTask,
    POSSubTask,
)
from ..utils.logger import general_logger


class HeadError(Exception):
    """Custom exception for head-related errors."""

    pass


def HeadFactory(st: SubTask, *args, **kwargs) -> "BaseHead":
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
            return ClassificationHead(
                num_classes=st.num_classes,
                class_weights=st.class_weights,
                *args,
                **kwargs,
            )
        elif isinstance(st, MultiLabelClassificationSubTask):
            return ClassificationHead(
                num_classes=st.num_classes,
                num_labels=st.num_labels if st.num_labels is not None else 2,
                class_weights=st.class_weights,
                *args,
                **kwargs,
            )
        elif isinstance(st, POSSubTask):
            return TokenClassificationHead(
                num_classes=st.num_classes,
                class_weights=st.class_weights,
                *args,
                **kwargs,
            )
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

    def forward(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
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
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Common layers
        self.dense = nn.Linear(input_dimension, hidden_dimension)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.out_proj = nn.Linear(hidden_dimension, num_classes * num_labels)

        # Store dimensions
        self.num_classes = num_classes
        self.num_labels = num_labels

        # Use CrossEntropyLoss for both cases
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        # Set up metrics based on task type
        if num_labels > 1:  # Multi-label case
            self.metrics = {
                "f1": F1Score(
                    num_classes=num_classes,
                    num_labels=num_labels,
                    task="multilabel",
                    average="macro",
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
                    average="macro",
                ),
                "acc": Accuracy(
                    task="binary" if num_classes == 2 else "multiclass",
                    num_classes=num_classes,
                ),
            }

    def forward(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        try:
            batch_size = y.shape[0]

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
            predictions = torch.argmax(
                logits, dim=-1
            )  # Use last dimension for class prediction

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
        **kwargs,
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_dimension, num_classes)
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

        self.metrics = {
            "f1": F1Score(num_classes=num_classes, task="multiclass", average="macro"),
            "acc": Accuracy(task="multiclass", num_classes=num_classes),
        }

        general_logger.info(
            f"Initialized TokenClassificationHead with {num_classes} classes"
        )

    def forward(
        self, X: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        try:
            # Process sequence
            sequence_output = self.dropout(X)
            logits = self.classifier(sequence_output)

            # Compute loss
            loss = self.loss(logits.view(-1, self.num_classes), y.view(-1))

            # Mask padding tokens for metrics
            mask = torch.where(y != -100, 1, 0)
            logits = torch.masked_select(
                logits, mask.unsqueeze(-1).expand(logits.size()) == 1
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
