"""Checkpoint management module for MTL model.

This module handles saving, loading, and managing model checkpoints,
including best model tracking and checkpoint rotation.
"""

from typing import Dict, Optional, Union, Any
import torch
from pathlib import Path
import json
import shutil
import time
from dataclasses import dataclass
from collections import deque

from ..utils.logger import general_logger
from ..utils.enums import Split


@dataclass
class CheckpointMetadata:
    """Container for checkpoint metadata.

    Attributes:
        epoch: Training epoch number
        global_step: Global training step
        train_loss: Training loss
        val_loss: Validation loss
        metrics: Dictionary of metrics
        timestamp: When checkpoint was created
    """
    epoch: int
    global_step: int
    train_loss: float
    val_loss: float
    metrics: Dict[str, float]
    timestamp: float


class CheckpointError(Exception):
    """Custom exception for checkpoint-related errors."""
    pass


class CheckpointManager:
    """Manages model checkpoints.

    This class handles saving and loading checkpoints, including
    maintaining best models and checkpoint rotation.

    Attributes:
        save_dir: Directory for saving checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        checkpoint_name: Base name for checkpoint files
        save_best_only: Whether to save only best models
        best_metric: Name of metric to track for best model
        minimize_metric: Whether metric should be minimized
    """

    def __init__(
            self,
            save_dir: Union[str, Path],
            max_checkpoints: int = 5,
            checkpoint_name: str = "model",
            save_best_only: bool = False,
            best_metric: str = "val_loss",
            minimize_metric: bool = True
    ):
        """Initialize checkpoint manager.

        Args:
            save_dir: Directory to save checkpoints in
            max_checkpoints: Maximum number of checkpoints to keep
            checkpoint_name: Base name for checkpoint files
            save_best_only: Whether to save only best models
            best_metric: Metric to track for best model
            minimize_metric: Whether metric should be minimized
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.checkpoint_name = checkpoint_name
        self.save_best_only = save_best_only
        self.best_metric = best_metric
        self.minimize_metric = minimize_metric

        # Track checkpoints
        self.checkpoints = deque(maxlen=max_checkpoints)
        self.best_checkpoint: Optional[Path] = None
        self.best_metric_value = float('inf') if minimize_metric else float('-inf')

        # Load existing checkpoints if any
        self._load_existing_checkpoints()

        general_logger.info(
            f"Initialized checkpoint manager in {save_dir} "
            f"(max_checkpoints={max_checkpoints}, save_best_only={save_best_only})"
        )

    def _load_existing_checkpoints(self) -> None:
        """Load information about existing checkpoints."""
        try:
            # Find all checkpoint files
            checkpoint_files = sorted(
                self.save_dir.glob(f"{self.checkpoint_name}*.pt")
            )

            for checkpoint_file in checkpoint_files:
                metadata_file = checkpoint_file.with_suffix('.json')
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                    # Update best checkpoint if applicable
                    if self.best_metric in metadata['metrics']:
                        metric_value = metadata['metrics'][self.best_metric]
                        if self._is_better_metric(metric_value):
                            self.best_checkpoint = checkpoint_file
                            self.best_metric_value = metric_value

                    self.checkpoints.append(checkpoint_file)

            general_logger.info(
                f"Found {len(self.checkpoints)} existing checkpoints"
            )

        except Exception as e:
            raise CheckpointError(f"Failed to load existing checkpoints: {str(e)}")

    def _is_better_metric(self, new_value: float) -> bool:
        """Check if new metric value is better than current best.

        Args:
            new_value: New metric value to compare

        Returns:
            Whether new value is better
        """
        if self.minimize_metric:
            return new_value < self.best_metric_value
        return new_value > self.best_metric_value

    def _save_metadata(
            self,
            path: Path,
            metadata: CheckpointMetadata
    ) -> None:
        """Save checkpoint metadata to JSON file.

        Args:
            path: Path to save metadata
            metadata: Metadata to save
        """
        try:
            metadata_dict = {
                'epoch': metadata.epoch,
                'global_step': metadata.global_step,
                'train_loss': metadata.train_loss,
                'val_loss': metadata.val_loss,
                'metrics': metadata.metrics,
                'timestamp': metadata.timestamp
            }

            with open(path.with_suffix('.json'), 'w') as f:
                json.dump(metadata_dict, f, indent=2)

        except Exception as e:
            raise CheckpointError(f"Failed to save metadata: {str(e)}")

    def save(
            self,
            model: torch.nn.Module,
            optimizer: Optional[torch.optim.Optimizer],
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
            metadata: CheckpointMetadata
    ) -> Optional[Path]:
        """Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optional optimizer to save
            scheduler: Optional scheduler to save
            metadata: Checkpoint metadata

        Returns:
            Path to saved checkpoint if saved, None otherwise

        Raises:
            CheckpointError: If saving fails
        """
        try:
            # Check if we should save
            metric_value = metadata.metrics.get(self.best_metric)
            if self.save_best_only and metric_value is not None:
                if not self._is_better_metric(metric_value):
                    return None

            # Create checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metadata': {
                    'epoch': metadata.epoch,
                    'global_step': metadata.global_step,
                    'metrics': metadata.metrics
                }
            }

            # Generate checkpoint path
            checkpoint_path = self.save_dir / (
                f"{self.checkpoint_name}_epoch{metadata.epoch:03d}.pt"
            )

            # Save checkpoint and metadata
            torch.save(checkpoint, checkpoint_path)
            self._save_metadata(checkpoint_path, metadata)

            # Update checkpoint tracking
            self.checkpoints.append(checkpoint_path)

            # Update best checkpoint if applicable
            if metric_value is not None and self._is_better_metric(metric_value):
                if self.best_checkpoint is not None:
                    old_best = self.best_checkpoint
                    if old_best != checkpoint_path:
                        shutil.copy(checkpoint_path, self.best_checkpoint.parent / 'best.pt')
                else:
                    shutil.copy(checkpoint_path, self.save_dir / 'best.pt')
                self.best_checkpoint = checkpoint_path
                self.best_metric_value = metric_value

            # Clean up old checkpoints if necessary
            while len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.popleft()
                if old_checkpoint != self.best_checkpoint:
                    old_checkpoint.unlink()
                    old_checkpoint.with_suffix('.json').unlink()

            general_logger.info(f"Saved checkpoint to {checkpoint_path}")
            return checkpoint_path

        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {str(e)}")

    def load(
            self,
            path: Optional[Union[str, Path]] = None,
            load_best: bool = False,
            map_location: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load checkpoint.

        Args:
            path: Path to checkpoint to load, or None for latest
            load_best: Whether to load best checkpoint
            map_location: Optional device to map tensors to

        Returns:
            Loaded checkpoint dictionary

        Raises:
            CheckpointError: If loading fails
        """
        try:
            if load_best:
                if self.best_checkpoint is None:
                    raise CheckpointError("No best checkpoint available")
                path = self.best_checkpoint
            elif path is None:
                if not self.checkpoints:
                    raise CheckpointError("No checkpoints available")
                path = self.checkpoints[-1]
            else:
                path = Path(path)

            if not path.exists():
                raise CheckpointError(f"Checkpoint not found: {path}")

            checkpoint = torch.load(path, map_location=map_location)
            general_logger.info(f"Loaded checkpoint from {path}")
            return checkpoint

        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {str(e)}")

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        return self.checkpoints[-1] if self.checkpoints else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint.

        Returns:
            Path to best checkpoint or None if no best checkpoint exists
        """
        return self.best_checkpoint

    def cleanup(self) -> None:
        """Remove all checkpoints."""
        try:
            for checkpoint in self.checkpoints:
                checkpoint.unlink()
                checkpoint.with_suffix('.json').unlink()
            self.checkpoints.clear()
            self.best_checkpoint = None
            self.best_metric_value = float('inf') if self.minimize_metric else float('-inf')
            general_logger.info("Removed all checkpoints")

        except Exception as e:
            raise CheckpointError(f"Failed to cleanup checkpoints: {str(e)}")