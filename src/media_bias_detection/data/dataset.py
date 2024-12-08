"""Dataset handling module for MTL model.

This module provides dataset classes for handling different types of data loading
and batch generation for the MTL training process.
"""

from typing import List, Dict, Iterator, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from dataclasses import dataclass
from collections import defaultdict

from media_bias_detection.utils.logger import general_logger
from media_bias_detection.utils.enums import Split
from .task import SubTask

#data class thing is not han

@dataclass
class BatchData:
    """Container for batch data.

    Attributes:
        input_ids: Token IDs from tokenizer
        attention_mask: Attention mask for padding
        labels: Target labels
        subtask_id: ID of the subtask this batch belongs to
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    subtask_id: int


class SubTaskDataset(Dataset):
    """Dataset class for a single SubTask.

    This dataset handles the loading and iteration over data for a specific subtask,
    with support for shuffling and automatic reset.

    Attributes:
        split: The data split (TRAIN/DEV/TEST)
        subtask: The subtask this dataset is for
        observations: List of indices into the data
        _counter: Internal counter for iteration
        cache: Optional cache for frequently accessed items
    """

    def __init__(
            self,
            subtask: SubTask,
            split: Split,
            cache_size: int = 100
    ) -> None:
        """Initialize the dataset.

        Args:
            subtask: SubTask instance containing the data
            split: Which data split to use
            cache_size: Number of items to keep in memory cache
        """
        general_logger.info(f"Initializing dataset for subtask {subtask.id} with split {split}")

        if not subtask.processed:
            raise RuntimeError(f"Subtask {subtask.id} must be processed before creating dataset")

        self.split = split
        self.subtask = subtask
        self.observations: List[int] = []
        self._counter: int = 0
        self._cache: Dict[int, BatchData] = {}
        self._cache_size = cache_size
        self._reset()

    def __len__(self) -> int:
        """Get number of items in dataset."""
        return len(self.observations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item to get

        Returns:
            BatchData containing the item data

        Raises:
            IndexError: If index is out of bounds
        """
        try:
            if self._counter >= len(self.observations):
                self._reset()

            i = self.observations[self._counter]

            # Check cache first
            if i in self._cache:
                self._counter += 1
                return self._cache[i]

            # Load and process item
            x = self.subtask.get_X(split=self.split)[i]
            masks = self.subtask.get_att_mask(split=self.split)[i]
            y = self.subtask.get_Y(split=self.split)[i]

            batch_data = BatchData(
                input_ids=x,
                attention_mask=masks,
                labels=y,
                subtask_id=self.subtask.id
            )

            # Update cache
            if len(self._cache) >= self._cache_size:
                # Remove oldest item
                del self._cache[next(iter(self._cache))]
            self._cache[i] = batch_data

            self._counter += 1
            return x, masks, y, self.subtask.id

        except Exception as e:
            general_logger.error(f"Error retrieving item {idx} from dataset: {str(e)}")
            raise

    def _reset(self) -> None:
        """Reset the dataset state and shuffle observations."""
        general_logger.info(f"Resetting dataset for subtask {self.subtask.id}")
        self.observations = list(range(len(self.subtask.get_X(split=self.split))))
        np.random.shuffle(self.observations)
        self._counter = 0
        self._cache.clear()


class BatchList:
    """Wrapper around dataloaders for continuous batch generation.

    This class provides an infinite stream of batches by automatically resetting
    exhausted dataloaders. It includes support for dynamic batch sizing and
    memory-efficient data loading.

    Attributes:
        sub_batch_size: Size of each sub-batch
        datasets: Mapping of subtask IDs to datasets
        dataloaders: Mapping of subtask IDs to dataloaders
        iter_dataloaders: Mapping of subtask IDs to dataloader iterators
    """

    def __init__(
            self,
            subtask_list: List[SubTask],
            sub_batch_size: int,
            split: Split = Split.TRAIN,
            num_workers: int = 0,
            pin_memory: bool = True
    ) -> None:
        """Initialize BatchList.

        Args:
            subtask_list: List of subtasks to create batches for
            sub_batch_size: Size of each sub-batch
            split: Which data split to use
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory in GPU training
        """
        general_logger.info(
            f"Creating BatchList with {len(subtask_list)} subtasks, "
            f"batch size {sub_batch_size}"
        )

        self.sub_batch_size = sub_batch_size
        self.split = split

        # Initialize datasets and dataloaders
        self.datasets = {
            str(st.id): SubTaskDataset(subtask=st, split=split)
            for st in subtask_list
        }

        self.dataloaders = {
            st_id: DataLoader(
                dataset,
                batch_size=self.sub_batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            for st_id, dataset in self.datasets.items()
        }

        self.iter_dataloaders = {
            st_id: iter(dl)
            for st_id, dl in self.dataloaders.items()
        }

        # Statistics tracking
        self._batch_counts = defaultdict(int)

    def __next__(self) -> List[BatchData]:
        """Get next batch of sub-batches.

        Returns:
            List of BatchData, one for each task

        Raises:
            RuntimeError: If batch generation fails
        """
        try:
            data = []
            items = list(self.iter_dataloaders.items())
            random.shuffle(items)

            for st_id, dl in items:
                try:
                    batch = next(dl)
                except StopIteration:
                    # Reset iterator and try again
                    self.iter_dataloaders[st_id] = iter(self.dataloaders[st_id])
                    batch = next(self.iter_dataloaders[st_id])

                data.append(batch)
                self._batch_counts[st_id] += 1

            general_logger.debug(f"Generated batch with {len(data)} sub-batches")
            return data

        except Exception as e:
            general_logger.error(f"Error generating batch: {str(e)}")
            raise RuntimeError(f"Batch generation failed: {str(e)}")

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about batches generated.

        Returns:
            Dictionary mapping subtask IDs to number of batches generated
        """
        return dict(self._batch_counts)


class BatchListEvalTest(BatchList):
    """BatchList variant for evaluation and testing.

    This class modifies the batch generation behavior to stop when any task's
    data is exhausted, making it suitable for evaluation and testing where
    we need to process all data exactly once.
    """

    def __next__(self) -> Optional[List[BatchData]]:
        """Get next batch of sub-batches, or None if any task is exhausted.

        Returns:
            List of BatchData or None if iteration is complete
        """
        try:
            data = []

            for st_id, dl in self.iter_dataloaders.items():
                try:
                    batch = next(dl)
                    data.append(batch)
                except StopIteration:
                    return None

            return data

        except Exception as e:
            general_logger.error(f"Error in evaluation batch generation: {str(e)}")
            raise RuntimeError(f"Evaluation batch generation failed: {str(e)}")

    def __len__(self) -> int:
        """Get total number of batches needed for full evaluation."""
        return min(len(dl) for dl in self.dataloaders.values())

    def reset(self) -> None:
        """Reset all iterators for a new evaluation pass."""
        general_logger.info("Resetting evaluation batch iterators")
        self.iter_dataloaders = {
            st_id: iter(dl)
            for st_id, dl in self.dataloaders.items()
        }