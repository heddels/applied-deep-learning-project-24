#TO DO: muss mit restlichen modulen abgeglichen werden

"""Module for creating instantiating the appropriate model defined by the task list only."""

from typing import List

import torch

from ..data.dataset import BatchList, BatchListEvalTest
from ..model.model import Model
from ..utils.enums import Split


def ModelFactory(
        task_list: List,
        sub_batch_size: int,
        eval_batch_size: int,
        pretrained_path: str = None,
        *args,
        **kwargs
):
    """Create model and return it along with dataloaders."""
    # Get all subtasks from task list
    subtask_list = [st for t in task_list for st in t.subtasks_list]

    # Verify data is processed
    for st in subtask_list:
        assert st.processed, "Data must be loaded at this point."

    # Create model
    model = Model(stl=subtask_list, **kwargs)

    if pretrained_path is not None:
        model = load_pretrained_weights(model, pretrained_path=pretrained_path)

    # Move model to appropriate device
    model.to(model.device)

    # Create dataloaders
    batch_list_train = BatchList(
        subtask_list=subtask_list,
        sub_batch_size=sub_batch_size,
        split=Split.TRAIN
    )

    batch_list_dev = BatchList(
        subtask_list=subtask_list,
        sub_batch_size=eval_batch_size,
        split=Split.DEV
    )

    batch_list_eval = BatchListEvalTest(
        subtask_list=subtask_list,
        sub_batch_size=sub_batch_size,
        split=Split.DEV
    )

    batch_list_test = BatchListEvalTest(
        subtask_list=subtask_list,
        sub_batch_size=sub_batch_size,
        split=Split.TEST
    )

    return model, batch_list_train, batch_list_dev, batch_list_eval, batch_list_test


def save_head_initializations(model):
    """Save weight initialization of the head. This method will not be called anymore.
     It's only for the initial saving of weight inits for all tasks."""
    for head_name in model.heads.keys():
        torch.save(model.heads[head_name].state_dict(), 'model_files/heads/' + head_name + '_init.pth')


def load_head_initializations(model):
    """Load fixed weight initialization for each head in order to ensure reproducibility."""
    for head_name in model.heads.keys():
        weights_path = 'model_files/heads/' + head_name + '_init.pth'
        head_weights = torch.load(weights_path)
        model.heads[head_name].load_state_dict(head_weights, strict=True)


def load_pretrained_weights(model, pretrained_path):
    """Load the weights of a pretrained model."""
    weight_dict = torch.load(pretrained_path)
    model.load_state_dict(weight_dict, strict=False)
    return model
