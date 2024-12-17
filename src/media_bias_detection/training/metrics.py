"""Metrics tracking and computation module.

This module tracks how well the model is performing during training:

1. Metric Types:
   Model Performance       Training Progress
   ├─ Accuracy            ├─ Losses
   ├─ F1 Score           └─ Learning Rate
   └─ Task-specific

2. Features:
   - Running averages for stable tracking
   - Separate tracking per task
   - History saving/loading
   - Best value tracking

3. Data Organization:
   Split (train/dev/test)
   └─ Task
      └─ Metrics
         ├─ Accuracy
         ├─ Loss
         └─ F1 Score
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np

from media_bias_detection.utils.enums import Split
from ..utils.logger import general_logger


class MetricError(Exception):
    """Custom exception for metric-related errors."""

    pass


class AverageMeter:
    """Tracks running average of a metric.

    Attributes:
        name: Name of the metric
        values: List of recorded values
    """

    def __init__(self, name: str):
        self.name = name
        self.values: List[float] = []

    def mean_last_k(self, k: int = 10) -> float:
        """Calculate mean of last k values.

        Args:
            k: Number of last values to average

        Returns:
            Mean of last k values or NaN if not enough values
        """
        if k < 1:
            general_logger.warning(f"Invalid k value ({k}) for metric {self.name}")
            return float("NaN")

        vals = self.values[-k:] if self.values else []
        if len(vals) < k:
            return float("NaN")

        return float(np.mean(vals))

    def mean_all(self) -> float:
        """Calculate mean of all values.

        Returns:
            Mean of all values or NaN if no values
        """
        if not self.values:
            return float("NaN")

        return float(np.mean(self.values))

    def update(self, value: float) -> None:
        """Add a new value to history.

        Args:
            value: Value to add
        """
        try:
            self.values.append(float(value))
        except (TypeError, ValueError) as e:
            general_logger.warning(f"Invalid value for metric {self.name}: {str(e)}")


class Tracker:
    """Tracks metrics and losses across training_baseline.

    This class manages metrics and losses for different splits
    and tasks, providing logging and analysis capabilities.

    Attributes:
        metrics: Nested dictionary of metrics for each split/task
        losses: Nested dictionary of losses for each split/task
        combined_losses: Dictionary of combined losses per split
        logger: Logger instance
    """

    def __init__(self, heads: Dict, logger: Any):
        """Initialize tracker.

        Args:
            heads: Dictionary of model heads
            logger: Logger instance
        """
        try:
            self.metrics = self._init_metrics(heads)
            self.losses, self.combined_losses = self._init_losses(heads)
            self.logger = logger

            # Track best metrics
            self.best_metrics: Dict[str, float] = {}

            general_logger.info("Initialized metric tracker")

        except Exception as e:
            raise MetricError(f"Failed to initialize tracker: {str(e)}")

    def _init_metrics(self, heads: Dict) -> Dict:
        """Initialize metric tracking structures.

        Args:
            heads: Dictionary of model heads

        Returns:
            Initialized metrics dictionary
        """
        try:
            metrics = {}
            for split in Split:
                metrics[split] = {
                    st_id: {
                        m: AverageMeter(name=f"{st_id}_{split.value}_{m}")
                        for m in head.metrics.keys()
                    }
                    for st_id, head in heads.items()
                }
            return metrics

        except Exception as e:
            raise MetricError(f"Failed to initialize metrics: {str(e)}")

    def _init_losses(self, heads: Dict) -> tuple:
        """Initialize loss tracking structures.

        Args:
            heads: Dictionary of model heads

        Returns:
            Tuple of (loss_dict, combined_loss_dict)
        """
        try:
            # Task-specific losses
            losses = {}
            for split in Split:
                losses[split] = {
                    st_id: AverageMeter(name=f"{st_id}_{split.value}_loss")
                    for st_id in heads.keys()
                }

            # Combined losses
            combined_losses = {
                split: AverageMeter(name=f"combined_{split.value}_loss")
                for split in Split
            }

            return losses, combined_losses

        except Exception as e:
            raise MetricError(f"Failed to initialize losses: {str(e)}")

    def __str__(self) -> str:
        """Return string representation of current training_baseline state."""
        try:
            # Get both training_baseline loss and metrics
            train_loss = self.combined_losses[Split.TRAIN].mean_last_k(1)
            metrics_str = []

            # Add training_baseline loss if available
            if not np.isnan(train_loss):
                metrics_str.append(f"Loss: {train_loss:.4f}")

            # Add first available metric if exists
            # This assumes your metrics dictionary structure from _init_metrics
            for st_id, st_metrics in self.metrics[Split.TRAIN].items():
                for metric_name, metric_meter in st_metrics.items():
                    metric_val = metric_meter.mean_last_k(1)
                    if not np.isnan(metric_val):
                        metrics_str.append(f"{metric_name}: {metric_val:.4f}")
                    break  # Just take first metric for conciseness
                break  # Just take first subtask for conciseness

            if metrics_str:
                return " | ".join(metrics_str)
            return "Initializing..."

        except Exception as e:
            general_logger.warning(f"Failed to create string representation: {str(e)}")
            return "No metrics available"

    def update_metric(
        self, split: Split, st_id: str, metric: str, value: float
    ) -> None:
        """Update a specific metric value.

        Args:
            split: Data split
            st_id: Subtask ID
            metric: Metric name
            value: New value
        """
        try:
            self.metrics[split][st_id][metric].update(value)

            # Track best metrics for validation
            if split == Split.DEV:
                metric_key = f"{st_id}_{metric}"
                current_value = value
                if (
                    metric_key not in self.best_metrics
                    or current_value > self.best_metrics[metric_key]
                ):
                    self.best_metrics[metric_key] = current_value

        except Exception as e:
            raise MetricError(f"Failed to update metric: {str(e)}")

    def update_loss(self, split: Split, st_id: str, value: float) -> None:
        """Update a specific loss value.

        Args:
            split: Data split
            st_id: Subtask ID
            value: New loss value
        """
        try:
            self.losses[split][st_id].update(value)
        except Exception as e:
            raise MetricError(f"Failed to update loss: {str(e)}")

    def update_combined_loss(self, split: Split, value: float) -> None:
        """Update combined loss for a split.

        Args:
            split: Data split
            value: New loss value
        """
        try:
            self.combined_losses[split].update(value)
        except Exception as e:
            raise MetricError(f"Failed to update combined loss: {str(e)}")

    def get_last_st_loss(self, split: Split, st_id: str, k: int) -> float:
        """Get mean of last k loss values for a subtask.

        Args:
            split: Data split
            st_id: Subtask ID
            k: Number of values to average

        Returns:
            Mean loss value
        """
        try:
            return self.losses[split][st_id].mean_last_k(k=k)
        except Exception as e:
            raise MetricError(f"Failed to get subtask loss: {str(e)}")

    def get_last_st_metric(self, split: Split, st_id: str, k: int) -> float:
        """Get mean of last k metric values for a subtask.

        Args:
            split: Data split
            st_id: Subtask ID
            k: Number of values to average

        Returns:
            Mean metric value
        """
        try:
            # Get first metric as representative
            first_metric = next(iter(self.metrics[split][st_id]))
            return self.metrics[split][st_id][first_metric].mean_last_k(k=k)
        except Exception as e:
            raise MetricError(f"Failed to get subtask metric: {str(e)}")

    def log(
        self, splits: List[Split], additional_payload: Optional[Dict[str, float]] = None
    ) -> None:
        """Log metrics and losses.

        Args:
            splits: List of splits to log
            additional_payload: Optional additional values to log
        """
        try:
            out: Dict[str, float] = additional_payload or {}

            for split in splits:
                # Check if we have any values before trying to log
                if not any(
                    any(m.values for m in d.values())
                    for d in self.metrics[split].values()
                ):
                    general_logger.warning(
                        f"No metrics recorded for split {split}, skipping logging"
                    )
                    continue
                # For training_baseline and validation, log last values
                if split in [Split.DEV, Split.TRAIN]:
                    # Log metrics
                    metrics = {
                        m.name: m.mean_last_k(1)
                        for d in self.metrics[split].values()
                        for m in d.values()
                    }
                    # Log losses
                    combined_loss = self.combined_losses[split].mean_last_k(1)
                    losses = {
                        v.name: v.mean_last_k(1) for v in self.losses[split].values()
                    }
                # For test and eval, log means
                else:
                    # Log metrics
                    metrics = {
                        m.name: m.mean_all()
                        for d in self.metrics[split].values()
                        for m in d.values()
                    }
                    # Log losses
                    combined_loss = self.combined_losses[split].mean_all()
                    losses = {v.name: v.mean_all() for v in self.losses[split].values()}

                out.update(metrics)
                out[f"combined_{split.value}_loss"] = combined_loss
                out.update(losses)

            # Log to wandb and local logger
            self.logger.log(out)

        except Exception as e:
            raise MetricError(f"Failed to log metrics: {str(e)}")

    def save_history(self, path: Union[str, Path]) -> None:
        """Save complete metric history.

        Args:
            path: Path to save history
        """
        try:
            path = Path(path)
            history = {
                "metrics": {
                    split.value: {
                        st_id: {
                            metric: meter.get_history()
                            for metric, meter in st_metrics.items()
                        }
                        for st_id, st_metrics in split_metrics.items()
                    }
                    for split, split_metrics in self.metrics.items()
                },
                "losses": {
                    split.value: {
                        st_id: meter.get_history()
                        for st_id, meter in split_losses.items()
                    }
                    for split, split_losses in self.losses.items()
                },
                "combined_losses": {
                    split.value: meter.get_history()
                    for split, meter in self.combined_losses.items()
                },
                "best_metrics": self.best_metrics,
            }

            with open(path, "w") as f:
                json.dump(history, f, indent=2)

            general_logger.info(f"Saved metric history to {path}")

        except Exception as e:
            raise MetricError(f"Failed to save history: {str(e)}")

    def load_history(self, path: Union[str, Path]) -> None:
        """Load metric history.

        Args:
            path: Path to load history from
        """
        try:
            path = Path(path)
            with open(path) as f:
                history = json.load(f)

            # Restore metrics
            for split_name, split_metrics in history["metrics"].items():
                split = Split(split_name)
                for st_id, st_metrics in split_metrics.items():
                    for metric, values in st_metrics.items():
                        for value in values:
                            self.metrics[split][st_id][metric].update(value)

            # Restore losses
            for split_name, split_losses in history["losses"].items():
                split = Split(split_name)
                for st_id, values in split_losses.items():
                    for value in values:
                        self.losses[split][st_id].update(value)

            # Restore combined losses
            for split_name, values in history["combined_losses"].items():
                split = Split(split_name)
                for value in values:
                    self.combined_losses[split].update(value)

            # Restore best metrics
            self.best_metrics = history["best_metrics"]

            general_logger.info(f"Loaded metric history from {path}")

        except Exception as e:
            raise MetricError(f"Failed to load history: {str(e)}")

    def reset(self) -> None:
        """Reset all metrics and losses."""
        try:
            # Reset metrics
            for split_metrics in self.metrics.values():
                for st_metrics in split_metrics.values():
                    for meter in st_metrics.values():
                        meter.reset()

            # Reset losses
            for split_losses in self.losses.values():
                for meter in split_losses.values():
                    meter.reset()

            # Reset combined losses
            for meter in self.combined_losses.values():
                meter.reset()

            general_logger.info("Reset all metrics and losses")

        except Exception as e:
            raise MetricError(f"Failed to reset metrics: {str(e)}")
