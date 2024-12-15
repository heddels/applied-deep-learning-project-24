"""This module can be executed to run a hyperparameter training_baseline job for multiple tasks, each task alone.
It is taken from the scripts/hyperparameter_tuning/hyperparameter_tuning.py file. of the MAGPIE repository."""

import wandb
from media_bias_detection.config.config import (
    hyper_param_dict,
    head_specific_lr,
    head_specific_max_epoch,
    head_specific_patience, MAX_NUMBER_OF_STEPS
)
from media_bias_detection.utils.enums import AggregationMethod, LossScaling
from media_bias_detection.data import all_subtasks
from media_bias_detection.data.task import Task
from media_bias_detection.training.training_utils import EarlyStoppingMode, Logger
from media_bias_detection.training.trainer import Trainer
from media_bias_detection.utils.common import set_random_seed
import media_bias_detection.data

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_wrapper():
    """Execute the wandb hyperparameter tuning job.

    Takes the (globally defined) tasks, instantiates a trainer for them.
    This function is passed as a callback to wandb.
    """
    EXPERIMENT_NAME = "hyperparam-tuning"
    wandb.init(project=EXPERIMENT_NAME)
    set_random_seed()

    our_config = {
        # Parameters to tune with wandb
        "sub_batch_size": wandb.config.sub_batch_size,
        "num_warmup_steps": wandb.config.num_warmup_steps,
        "dropout_prob": wandb.config.dropout_prob,
        # Fixed parameters
        "max_steps": 500,
        "eval_batch_size": 128,
        "initial_lr": 5e-5,
        "hidden_dimension": 768,
        "input_dimension": 768,
        "early_stopping_mode": EarlyStoppingMode.HEADS,
        "aggregation_method": AggregationMethod.MEAN,
        "loss_scaling": LossScaling.UNIFORM,
        "model_name": "hyperparameter_tuning",
        "pretrained_path": None,
        "resurrection": False,
        "head_specific_lr_dict": head_specific_lr,
        "head_specific_patience_dict": head_specific_patience,
        "head_specific_max_epoch_dict": head_specific_max_epoch,
        "logger": Logger(EXPERIMENT_NAME)
    }

    trainer = Trainer(task_list=task_wrapper, **our_config)
    trainer.fit()


if __name__ == "__main__":
    for st in [media_bias_detection.data.st_1_gwsd_128]: # change to all_subtasks, if it is working
        st.process()
        task_wrapper = [Task(task_id=st.id, subtasks_list=[st])]

        # wandb sweep
        sweep_config = {"method": "random",
                        "metric": {"name": "eval_f1", "goal": "maximize"},
                        "parameters": hyper_param_dict,
                        "early_terminate": { "type": "hyperband", "min_iter": 3},
                        "name": str(st)}
        sweep_id = wandb.sweep(sweep_config, project="hyperparam-tuning")

        wandb.agent(sweep_id, train_wrapper)
        wandb.finish()