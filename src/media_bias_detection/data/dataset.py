

"""Dataset handling module for MTL model.

This module provides dataset classes for handling different types of data loading
and batch generation for the MTL training_baseline process.
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
from media_bias_detection.utils.common import set_random_seed
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
    """

    def __init__(
            self,
            subtask: SubTask,
            split: Split,
    ) -> None:
        """Initialize the dataset.

        Args:
            subtask: SubTask instance containing the data
            split: Which data split to use
        """
        general_logger.info(f"Initializing dataset for subtask {subtask.id} with split {split}")

        if not subtask.processed:
            raise RuntimeError(f"Subtask {subtask.id} must be processed before creating dataset")

        self.split = split
        self.subtask = subtask
        self.observations: List[int] = []
        self._counter: int = 0
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

            self._counter += 1
            return x, masks, y, self.subtask.id

        except Exception as e:
            general_logger.error(f"Error retrieving item {idx} from dataset: {str(e)}")
            raise

    def _reset(self) -> None:
        """Reset the dataset state and shuffle observations."""
        general_logger.info(f"Resetting dataset for subtask {self.subtask.id}")
        self.observations = [i for i in range(len(self.subtask.get_X(split=self.split)))]
        set_random_seed()
        np.random.shuffle(self.observations)  # Not a real 'reshuffling' as it will always arrange same.
        self._counter = 0


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
            pin_memory: Whether to pin memory in GPU training_baseline
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

    def _reset(self):
        """Reset this BatchListEvalTest."""
        self.iter_dataloaders = {f"{st_id}": iter(dl) for st_id, dl in self.dataloaders.items()}


class BatchListEvalTest:
    """A BatchListEvalTest is a wrapper around dataloaders for each subtask."""

    def __init__(self, subtask_list: List[SubTask], sub_batch_size, split=Split.TRAIN):
        self.sub_batch_size = sub_batch_size
        self.datasets = {f"{st.id}": SubTaskDataset(subtask=st, split=split) for st in subtask_list}
        self.dataloaders = {
            f"{st_id}": DataLoader(ds, batch_size=self.sub_batch_size) for st_id, ds in self.datasets.items()
        }
        self.iter_dataloaders = {f"{st_id}": iter(dl) for st_id, dl in self.dataloaders.items()}

    def __len__(self):
        return min(len(dl) for dl in self.dataloaders.values())

    def _reset(self): # Add this method matching the original
        self.iter_dataloaders = {f"{st_id}": iter(dl) for st_id, dl in self.dataloaders.items()}