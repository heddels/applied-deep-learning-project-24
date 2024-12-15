"""Script for executing the pre-finetuning on the subtasks excluding BABE
with 500 Steps and 50 Warm Up Steps."""
from pathlib import Path

import wandb
from click import clear

# Import configuration and utilities
from media_bias_detection.utils.enums import Split, AggregationMethod, LossScaling
from media_bias_detection.utils.common import set_random_seed
from media_bias_detection.config.config import (
    head_specific_lr,
    head_specific_max_epoch,
    head_specific_patience
)
from media_bias_detection.training.training_utils import Logger, EarlyStoppingMode

from media_bias_detection.data import all_tasks, babe_10

from media_bias_detection.training.trainer import Trainer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    EXPERIMENT_NAME = "hyperparam-tuning"
    MODEL_NAME = "pre_finetuned_model"


    tasks = all_tasks
    all_tasks.remove(babe_10)

    for t in tasks:
        for st in t.subtasks_list:
            st.process()

    config = {
        "sub_batch_size": 32,
        "eval_batch_size": 128,
        "initial_lr": 4e-5,
        "dropout_prob": 0.1,
        "hidden_dimension": 768,
        "input_dimension": 768,
        "aggregation_method": AggregationMethod.PCGRAD_ONLINE,
        "early_stopping_mode": EarlyStoppingMode.HEADS,
        "loss_scaling": LossScaling.STATIC,
        "num_warmup_steps": 50,
        "pretrained_path": None,
        "resurrection": True,
        "model_name": MODEL_NAME,
        "max_steps": 500,
        "head_specific_lr_dict": head_specific_lr,
        "head_specific_patience_dict": head_specific_patience,
        "head_specific_max_epoch_dict": head_specific_max_epoch,
        "logger": Logger(EXPERIMENT_NAME),
    }

    set_random_seed()
    try:
        wandb.init(project=EXPERIMENT_NAME, name="all_tasks_pretraining_pconline")
        trainer = Trainer(task_list=tasks, **config)
        trainer.fit()
        trainer.save_model()

    except KeyboardInterrupt:

        print("\nTraining interrupted! Saving checkpoint...")

        # Save with simpler checkpoint system

        trainer.checkpoint_manager.save_checkpoint(

            model=trainer.model,

            step=-1,  # Special flag for interrupted training_baseline

            metrics={'interrupted': True}

        )

        print("Checkpoint saved!")

    except Exception as e:
        print(f"Pre-Finetuning failed with error: {str(e)}")
        raise e

    finally:
        wandb.finish()
        print("Pre-Finetuning completed successfully!")

if __name__ == "__main__":
    main()