"""Main training script."""

import wandb
import argparse

from media_bias_detection.training.trainer import Trainer, EarlyStoppingMode
from src.training.logger import Logger
from media_bias_detection.utils import Split, AggregationMethod, LossScaling
from media_bias_detection.utils import set_random_seed
from media_bias_detection.config.config import (
    head_specific_lr,
    head_specific_max_epoch,
    head_specific_patience
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default='baseline_check')
    parser.add_argument('--model_name', default='baseline_model')
    parser.add_argument('--task_family', default='media_bias')
    # Add other arguments as needed
    return parser.parse_args()


def get_training_config(experiment_name: str, model_name: str):
    """Get training configuration."""
    return {
        "sub_batch_size": 32,
        "eval_batch_size": 128,
        "initial_lr": 4e-5,
        "dropout_prob": 0.1,
        "hidden_dimension": 768,
        "input_dimension": 768,
        "aggregation_method": AggregationMethod.MEAN,
        "early_stopping_mode": EarlyStoppingMode.HEADS,
        "loss_scaling": LossScaling.STATIC,
        "num_warmup_steps": 10,
        "pretrained_path": None,
        "resurrection": True,
        "model_name": model_name,
        "head_specific_lr_dict": head_specific_lr,
        "head_specific_patience_dict": head_specific_patience,
        "head_specific_max_epoch_dict": head_specific_max_epoch,
        "logger": Logger(experiment_name),
    }


def main():
    """Main training function."""
    args = parse_args()

    # Get tasks for selected family
    selected_tasks = TASK_FAMILIES[args.task_family]

    # Process tasks
    for task in selected_tasks:
        for subtask in task.subtasks_list:
            subtask.process()

    # Initialize wandb
    wandb.init(
        project=args.experiment_name,
        name=args.model_name
    )

    # Get training config
    config = get_training_config(args.experiment_name, args.model_name)

    # Set random seed
    set_random_seed()

    # Initialize trainer and train
    trainer = Trainer(task_list=selected_tasks, **config)
    trainer.fit_debug(k=1)
    trainer.eval(split=Split.TEST)
    trainer.save_model()

    wandb.finish()


if __name__ == "__main__":
    main()