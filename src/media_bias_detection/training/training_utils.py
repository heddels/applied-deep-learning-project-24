"""Training utilities module for MTL training_baseline.

Contains utility classes for:
- Logging
- Early stopping

"""

import math
import os
import logging

from enum import Enum
from typing import Dict, List, Any, Optional

from media_bias_detection.utils.logger import general_logger
from media_bias_detection.utils.enums import Split

import wandb



class Logger:
    """Logger to keep track of metrics, losses and artifacts."""

    def __init__(self, experiment_name: str):
        PATH = "metric_logging/" + experiment_name
        os.makedirs(PATH, exist_ok=True)

        self.experiment_logfilename = PATH + "/train_data.log"
        experiment_logfile_handler = logging.FileHandler(filename=self.experiment_logfilename)
        experiment_logfile_formatter = logging.Formatter(fmt="%(message)s")
        experiment_logfile_handler.setFormatter(experiment_logfile_formatter)

        self.experiment_logger = logging.getLogger("experiment_logger")
        self.experiment_logger.addHandler(experiment_logfile_handler)
        self.experiment_logger.setLevel("INFO")

    def log(self, out: Dict[str, Any]) -> None:
        try:
            self.experiment_logger.info(out)
            wandb.log(out)
        except Exception as e:
            print(f"Logging failed: {str(e)}")


class EarlyStoppingMode(Enum):
    """Mode for early stopping behavior."""
    HEADS = "heads"  # Only stop heads
    BACKBONE = "backbone"  # Also stop backbone
    NONE = "none"  # No early stopping


class EarlyStopperSingle:
    """Early stopping tracker for a single model component."""

    def __init__(
            self,
            patience: int,
            min_delta: float,
            resurrection: bool,
            zombie_patience: int = 10
    ):
        """Initialize early stopping tracker.

        Args:
            patience: How many epochs to wait before stopping
            min_delta: Minimum change to count as improvement
            resurrection: Whether to allow resurrection
            zombie_patience: Patience for zombie state
        """
        self.patience = patience
        self.patience_zombie = zombie_patience
        self.min_delta = min_delta
        self.counter = 0
        self.counter_zombie = 0
        self.min_dev_loss = float('inf')
        self.min_dev_loss_zombie = float('inf')
        self.resurrection = resurrection

    def early_stop(self, dev_loss: float) -> bool:
        """Check if training_baseline should stop.

        Args:
            dev_loss: Current validation loss

        Returns:
            Whether to stop training_baseline
        """
        if math.isnan(dev_loss):
            return False

        if dev_loss < self.min_dev_loss:
            self.min_dev_loss = dev_loss
            self.counter = 0
        elif dev_loss > (self.min_dev_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def resurrect(self, dev_loss: float) -> bool:
        """Check if training_baseline should resume.

        Args:
            dev_loss: Current validation loss

        Returns:
            Whether to resume training_baseline
        """
        if math.isnan(dev_loss) or not self.resurrection:
            return False

        if dev_loss < self.min_dev_loss_zombie:
            self.min_dev_loss_zombie = dev_loss
            self.counter_zombie = 0
        elif dev_loss > self.min_dev_loss_zombie:
            self.counter_zombie += 1
            if self.counter_zombie >= self.patience_zombie:
                return True
        return False

    def reset(self) -> None:
        """Reset early stopping state."""
        self.counter_zombie = 0
        self.counter = 0
        self.min_dev_loss_zombie = float('inf')
        self.min_dev_loss = float('inf')


class EarlyStopper:
    """Container for managing multiple early stoppers."""

    def __init__(
            self,
            st_ids: List[str],
            mode: EarlyStoppingMode,
            patience: Dict[str, int],
            resurrection: bool,
            min_delta: float = 0
    ):
        """Initialize early stopping manager.

        Args:
            st_ids: List of subtask IDs
            mode: Early stopping mode
            patience: Dictionary of patience values per subtask
            resurrection: Whether to allow resurrection
            min_delta: Minimum improvement threshold
        """
        self.mode = mode
        self.early_stoppers = {
            st_id: EarlyStopperSingle(
                patience=patience[st_id],
                min_delta=min_delta,
                resurrection=resurrection
            )
            for st_id in st_ids
        }
        general_logger.info(
            f"Initialized early stopping manager with mode {mode}"
        )

    def early_stop(self, st_id: str, dev_loss: float) -> bool:
        """Check if specific task should stop."""
        return (
            False if self.mode == EarlyStoppingMode.NONE
            else self.early_stoppers[st_id].early_stop(dev_loss=dev_loss)
        )

    def resurrect(self, st_id: str, dev_loss: float) -> bool:
        """Check if specific task should resurrect."""
        return (
            False if self.mode == EarlyStoppingMode.NONE
            else self.early_stoppers[st_id].resurrect(dev_loss=dev_loss)
        )

    def reset_early_stopper(self, st_id: str) -> None:
        """Reset early stopper for specific task."""
        self.early_stoppers[st_id].reset()