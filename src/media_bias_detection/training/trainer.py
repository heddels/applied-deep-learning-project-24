"""Training module for MTL model.

Provides enhanced training_baseline functionality with:
- Comprehensive error handling
- Memory optimization
- Detailed logging
- Training efficiency improvements
"""
from typing import Dict, List, Optional, Any
from pathlib import Path
import gc
import time

import numpy as np
import psutil
import torch
import statistics as stats
from tqdm import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup

from ..model.model_factory import ModelFactory
from ..utils.enums import Split, LossScaling
from ..utils.logger import general_logger
from .metrics import Tracker
from .checkpoint_system import ModelCheckpoint
from .training_utils import Logger, EarlyStopper, EarlyStoppingMode

from ..model.gradient import GradientAggregator, AggregationMethod
from ..data.task import Task
from ..data.dataset import BatchData


class TrainerError(Exception):
    """Custom exception for training_baseline-related errors."""
    pass


class Trainer:
    """Enhanced trainer for MTL model.

    Features:
    - Automatic mixed precision training_baseline
    - Memory-optimized batch processing
    - Detailed progress tracking
    - Comprehensive error handling
    """

    def __init__(
            self,
            task_list: List[Task],
            initial_lr: float,
            model_name: str,
            max_steps: int,
            pretrained_path: Optional[str],
            sub_batch_size: int,
            eval_batch_size: int,
            early_stopping_mode,
            resurrection: bool,
            aggregation_method: AggregationMethod,
            loss_scaling: LossScaling,
            num_warmup_steps: int,
            head_specific_lr_dict: Dict[str, float],
            head_specific_patience_dict: Dict[str, int],
            head_specific_max_epoch_dict: Dict[str, int],
            logger: Logger,
            device: Optional[torch.device] = None,
            use_amp: bool = True,
            *args,
            **kwargs,
    ):
        """Initialize trainer with enhanced configuration."""
        try:
            self.logger = logger
            general_logger.info("Initializing trainer...")

            # Basic setup
            self.early_stopping_mode = early_stopping_mode
            self.loss_scaling = loss_scaling
            self.use_amp = use_amp and torch.cuda.is_available()
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.model, batch_list_train, batch_list_dev, batch_list_eval, batch_list_test = ModelFactory(
                task_list=task_list,
                sub_batch_size=sub_batch_size,
                eval_batch_size=eval_batch_size,
                pretrained_path=pretrained_path,
                *args,
                **kwargs,
            )
            self.batch_lists = {
                Split.TRAIN: batch_list_train,
                Split.DEV: batch_list_dev,
                Split.EVAL: batch_list_eval,
                Split.TEST: batch_list_test,
            }

            # shared backbone model optimizer
            self.lm_optimizer = torch.optim.AdamW(self.model.language_model.backbone.parameters(), lr=initial_lr)
            self.lm_lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer=self.lm_optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=max([len(dl) for dl in self.batch_lists[Split.TRAIN].dataloaders.values()])
                                   * stats.median(head_specific_max_epoch_dict.values()),
            )

            # task-specifics optimizers
            self.head_optimizers = {
                str(st_id): torch.optim.AdamW(head.parameters(), lr=head_specific_lr_dict[st_id])
                for st_id, head in self.model.heads.items()
            }
            self.head_lr_schedulers = {
                str(st_id): get_polynomial_decay_schedule_with_warmup(
                    optimizer=self.head_optimizers[st_id],
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=len(self.batch_lists[Split.TRAIN].dataloaders[st_id])
                                       * head_specific_max_epoch_dict[st_id],
                )
                for st_id in self.model.heads.keys()
            }

            # flags controlling stopping and resurrection
            self.task_alive_flags = {str(st_id): True for st_id in self.model.heads.keys()}
            self.task_zombie_flags = {str(st_id): False for st_id in self.model.heads.keys()}
            self.early_stopper = EarlyStopper(
                st_ids=self.model.heads.keys(),
                mode=self.early_stopping_mode,
                patience=head_specific_patience_dict,
                resurrection=resurrection,
            )

            # Initialize tracking components
            self.tracker = Tracker(heads=self.model.heads, logger=logger)
            self.GA = GradientAggregator(aggregation_method=aggregation_method)
            self.progress_bar = tqdm(total=len(self.model.heads), desc="Training Progress")
            self.model_name = model_name
            self.scaling_weights = {str(st.id): st.get_scaling_weight() for t in task_list for st in t.subtasks_list}
            self.max_steps = max_steps
            self.k = 50

            # Memory tracking
            self._last_memory_check = time.time()
            self._memory_check_interval = 60  # seconds

            #checkpoint initialization
            self.checkpoint_manager = ModelCheckpoint(
                save_dir=f"checkpoints/{model_name}",
                save_freq=50,
                monitor='combined_dev_loss',  # Monitor validation loss
            )

            general_logger.info(
                f"Trainer initialized successfully on {self.device}"
                f"{' with AMP' if self.use_amp else ''}"
            )

        except Exception as e:
            general_logger.error(f"Failed to initialize trainer: {str(e)}")
            raise TrainerError(f"Failed to initialize trainer: {str(e)}")

    def _optimize_memory(self) -> None:
        """Perform memory optimization."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            process = psutil.Process()
            memory_info = process.memory_info().rss / 1024 ** 2  # Convert to MB
            general_logger.debug(f"Current memory usage: {memory_info:.2f} MB")

        except Exception as e:
            general_logger.warning(f"Memory optimization failed: {str(e)}")

    def head_specific_optimization(self, st_id: str, lm_grads, scaling_weight):
        """Perform the optimization of a task-specific head.

        Args:
            st_id: The subtask id
            lm_grads: The LM gradients
            scaling_weight: The scaling weight of that subtask

        Returns:
            Dictionary with additional payload

        Raises:
            TrainerError: If optimization fails
        """
        try:
            general_logger.info(f"Optimizing task {st_id}, alive: {self.task_alive_flags[st_id]}")
            additional_payload = {}
            last_dev_loss = self.tracker.get_last_st_loss(split=Split.DEV, st_id=st_id, k=self.k)

            should_stop_now = (
                self.early_stopper.early_stop(st_id=st_id, dev_loss=last_dev_loss)
                if (self.task_alive_flags[st_id] or self.task_zombie_flags[st_id])
                else False
            )

            should_resurrect_now = (
                self.early_stopper.resurrect(st_id=st_id, dev_loss=last_dev_loss)
                if (not self.task_zombie_flags[st_id] and not self.task_alive_flags[st_id])
                else False
            )

            should_stay_zombie = (
                    not self.task_alive_flags[st_id] and
                    self.task_zombie_flags[st_id] and
                    not should_stop_now
            )

            # Handle task death
            if should_stop_now and self.task_alive_flags[st_id]:
                general_logger.info(f"Subtask {st_id} is now DEAD.")
                self.eval_st(split=Split.EVAL, st_id=st_id)
                self.tracker.log(splits=[Split.EVAL], additional_payload={st_id + "_STOPPED": 0})
                self.progress_bar.update()

            # Handle task resurrection
            elif should_resurrect_now and not self.task_zombie_flags[st_id]:
                general_logger.info(f"Subtask {st_id} is now ZOMBIE.")
                additional_payload[st_id + "_ZOMBIE"] = 0
                self.early_stopper.reset_early_stopper(st_id=st_id)

            # Handle zombie death
            elif should_stop_now and self.task_zombie_flags[st_id]:
                general_logger.info(f"Subtask {st_id} is now DEAD AGAIN.")
                additional_payload[st_id + "_DEAD_ZOMBIE"] = 0
                self.early_stopper.reset_early_stopper(st_id=st_id)

            # Update task states
            self.task_alive_flags[st_id] = self.task_alive_flags[st_id] and not (
                    should_stop_now or self.tracker.get_last_st_metric(split=Split.DEV, st_id=st_id, k=10) == 1
            )
            self.task_zombie_flags[st_id] = should_resurrect_now or should_stay_zombie

            # Optimize if task is alive or zombie
            optimize_task = self.task_alive_flags[str(st_id)] or self.task_zombie_flags[str(st_id)]
            if optimize_task:
                self.head_optimizers[st_id].step()
                self.head_lr_schedulers[st_id].step()

            # Update gradients
            if self.early_stopping_mode != EarlyStoppingMode.BACKBONE or optimize_task:
                self.GA.update(lm_grads, scaling_weight=scaling_weight)

            return additional_payload

        except Exception as e:
            raise TrainerError(f"Failed to do head specific optimization: {str(e)}")

    def backbone_optimization(self) -> Dict[str, Any]:
        """
        Perform the optimization of the backbone.

        This method is only called when mode is training_baseline.
        @return: A dictionary with additional payload containing the conflicting gradients ratio.
        """
        # Optimize the LM such that: we aggregate gradients from subtasks and set the final
        # gradient to the LM and subsequently optimize (only the LM)
        try:
            additional_payload = {}
            if any(self.task_alive_flags.values()):
                aggregated_gradients = self.GA.aggregate_gradients()
                self.model.language_model.set_grads(aggregated_gradients)
                self.lm_optimizer.step()
                self.lm_lr_scheduler.step()
            if self.GA.aggregation_method in [AggregationMethod.PCGRAD, AggregationMethod.PCGRAD_ONLINE]:
                conflicting_gradients_ratio = self.GA.get_conflicting_gradients_ratio()
                additional_payload["conflicting_gradients_ratio"] = conflicting_gradients_ratio
        except Exception as e:
            raise TrainerError(f"Failed to do backbone optimization: {str(e)}")
        return additional_payload

    def handle_batch(self, batch, split: Split = Split.TRAIN) -> Dict[str, Any]:
        """Handle a batch.

         (always) Pass a batch of sub_batches through the network.
         (in train-mode) For each sub_batch, accumulate the gradients of the LM.
         For each sub_batch and each st_id,
            - (in train-mode) accumulate the gradients of the respective head,
            - (always) accumulate the metric of the respective head,
            - (always) accumulate the loss of the respective head.
        (always) Log all metrics and losses to wandb.
         (in train-mode) After all sub_batches are processed, normalize the LM gradients and the head-specific gradients.
         (in train-mode) Then, perform the step of the lr_scheduler and the optimizer.

        @param batch: The batch containing sub-batches.
        @param split: The split (TRAIN, DEV, TEST)
        @return: A dictionary containing additional payload that needs to be logged.
        """
        try:
            training = split == Split.TRAIN
            losses = []
            additional_payloads = {}

            if training:
                self.GA.reset_accumulator()
                general_logger.debug("Reset gradient accumulator")

            for sub_batch in batch:
                if isinstance(sub_batch, BatchData):
                    X = sub_batch.input_ids
                    attention_masks = sub_batch.attention_mask
                    Y = sub_batch.labels
                    st_id = sub_batch.subtask_id
                else:
                    X, attention_masks, Y, st_id = sub_batch
                st_id_str = str(st_id.unique().item())

                general_logger.debug(f"Processing sub-batch for task {st_id_str}")

                # Forward pass and compute loss
                loss, metric_values, lm_grads = self._step(
                    (X, attention_masks, Y, st_id.unique()),
                    training=training
                )

                scaling_weight = (
                    self.scaling_weights[st_id_str]
                    if self.loss_scaling == LossScaling.STATIC
                    else 1.0
                )

                if training:
                    payload = self.head_specific_optimization(
                        st_id=st_id_str,
                        lm_grads=lm_grads,
                        scaling_weight=scaling_weight
                    )
                    additional_payloads.update(payload)

                # Update metrics and losses
                for metric, value in metric_values.items():
                    self.tracker.update_metric(split=split, st_id=st_id_str, metric=metric, value=value)
                self.tracker.update_loss(split=split, st_id=st_id_str, value=loss.item())
                losses.append(loss.item())

            if training:
                payload = self.backbone_optimization()
                additional_payloads.update(payload)

            mean_loss = np.mean(losses)
            self.tracker.update_combined_loss(split=split, value=mean_loss)
            general_logger.debug(f"Batch processed. Mean loss: {mean_loss:.4f}")

            return additional_payloads

        except Exception as e:
            general_logger.error(f"Failed to handle batch: {str(e)}")
            raise TrainerError(f"Batch processing failed: {str(e)}")

    def _step(self, batch, training: bool = True):
        """Perform a single training_baseline/evaluation step."""
        inputs = {
            "X": batch[0].to(self.device),
            "attention_masks": batch[1].to(self.device),
            "Y": batch[2].to(self.device),
            "st_id": batch[3]
        }

        try:
            if training:
                self.model.train()
                self.lm_optimizer.zero_grad()
                for optim in self.head_optimizers.values():
                    optim.zero_grad()

                loss, metric_values = self.model(**inputs)
                loss.backward()
                lm_gradients = self.model.language_model.get_grads()

            else:
                self.model.eval()
                with torch.no_grad():
                    loss, metric_values = self.model(**inputs)
                lm_gradients = None

            return loss, metric_values, lm_gradients

        except Exception as e:
            general_logger.error(f"Step execution failed: {str(e)}")
            raise TrainerError(f"Step execution failed: {str(e)}")

        finally:
            # Clean up to prevent memory leaks
            tensor_keys = [k for k, v in inputs.items() if isinstance(v, torch.Tensor)]
            for k in tensor_keys:
                del inputs[k]

    def fit_debug(self, k: int):
        """Fit for k iterations only to check if a model can process the data."""
        try:
            general_logger.info(f"Starting the debug training_baseline for {k} iterations")
            step = 0
            for _ in range(k):
                step += 1
                batch = next(self.batch_lists[Split.TRAIN])
                self.handle_batch(batch=batch, split=Split.TRAIN)
                # Evaluate on dev-batch
                batch = next(self.batch_lists[Split.DEV])
                self.handle_batch(batch=batch, split=Split.DEV)
        except Exception as e:
            general_logger.error(f"Debug training_baseline failed: {str(e)}")
            raise TrainerError(f"Debug training_baseline failed: {str(e)}")

    def fit(self):
        """Train the model."""
        try:
            general_logger.info(f"Starting training_baseline with MAX_STEPS: {self.max_steps}")
            general_logger.info(f"Initial task states: {self.task_alive_flags}")
            step = 0

            while step < self.max_steps:
                if not any(self.task_alive_flags.values()):
                    general_logger.info("No tasks remaining alive, stopping training_baseline")
                    break

                step += 1
                general_logger.debug(f"Starting step {step}")

                batch = next(self.batch_lists[Split.TRAIN])
                train_payload = self.handle_batch(batch=batch, split=Split.TRAIN)
                splits_to_log = [Split.TRAIN]

                if step % 3 == 0:
                    batch = next(self.batch_lists[Split.DEV])
                    dev_payload = self.handle_batch(batch=batch, split=Split.DEV)
                    train_payload.update(dev_payload)
                    splits_to_log.append(Split.DEV)

                self._update_progress()
                self.tracker.log(
                    splits=splits_to_log,
                    additional_payload=train_payload
                )

                # Add checkpoint saving every 50 steps
                if step % 50 == 0:  # Only save every 50 steps
                    metrics = {
                        'combined_dev_loss': self.tracker.combined_losses[Split.DEV].mean_last_k(1),
                        'step': step
                    }

                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        step=step,
                        metrics=metrics
                    )
                    general_logger.info(f"Saved checkpoint at step {step}")
                # Periodic memory optimization
                if step % 100 == 0:
                    self._optimize_memory()

            general_logger.info("Training completed")
            self.eval(split=Split.EVAL)

        except Exception as e:
            general_logger.error(f"Training failed: {str(e)}")
            raise TrainerError(f"Training failed: {str(e)}")

    def eval(self, split):
        """Evaluate the model."""
        try:
            general_logger.info(f"Starting evaluation on {split}")
            assert split in [Split.EVAL, Split.TEST]

            for st_id in self.batch_lists[split].iter_dataloaders.keys():
                self.eval_st(split=split, st_id=st_id)

            self.tracker.log(splits=[split])
            general_logger.info(f"Evaluation on {split} completed")

        except Exception as e:
            general_logger.error(f"Evaluation failed: {str(e)}")
            raise TrainerError(f"Evaluation failed: {str(e)}")

    def eval_st(self, split, st_id):
        """Evaluate on a specific subtask."""
        try:
            general_logger.debug(f"Evaluating subtask {st_id} on {split}")
            batch_list = self.batch_lists[split]
            batch_list._reset()
            idl = batch_list.iter_dataloaders[st_id]

            for batch in idl:
                _ = self.handle_batch(batch=[batch], split=split)

        except Exception as e:
            general_logger.error(f"Subtask evaluation failed: {str(e)}")
            raise TrainerError(f"Subtask evaluation failed: {str(e)}")

    def save_model(self):
        """Save the trained model."""
        try:
            path = Path("model_files")
            path.mkdir(exist_ok=True)
            model_path = path / f"{self.model_name}.pth"

            torch.save(self.model.state_dict(), model_path)
            general_logger.info(f"Model saved to {model_path}")

        except Exception as e:
            general_logger.error(f"Failed to save model: {str(e)}")
            raise TrainerError(f"Model saving failed: {str(e)}")

    def _update_progress(self):
        """Update progress bar."""
        try:
            desc = str(self.tracker)
            self.progress_bar.set_description(desc=desc)
            self.progress_bar.refresh()

        except Exception as e:
            general_logger.warning(f"Failed to update progress bar: {str(e)}")
