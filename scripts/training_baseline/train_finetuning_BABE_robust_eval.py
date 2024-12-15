"""Script for executing the finetuning and evaluation on BABE."""
from pathlib import Path

import wandb

# Import configuration and utilities
from media_bias_detection.utils.enums import Split, AggregationMethod, LossScaling
from media_bias_detection.utils.common import set_random_seed
from media_bias_detection.config.config import (
    head_specific_lr,
    head_specific_max_epoch,
    head_specific_patience
)
from media_bias_detection.training.training_utils import Logger, EarlyStoppingMode

from media_bias_detection.data import st_1_babe_10 as babe
from media_bias_detection.data.task import Task

from media_bias_detection.training.trainer import Trainer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    EXPERIMENT_NAME = "evaluation_robust"
    MODEL_NAME = "finetuned_babe_model"


    tasks = [Task(task_id=babe.id, subtasks_list=[babe])]

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
        "aggregation_method": AggregationMethod.MEAN,
        "early_stopping_mode": EarlyStoppingMode.HEADS,
        "loss_scaling": LossScaling.STATIC,
        "num_warmup_steps": 10,
        "pretrained_path": "model_files/pre_finetuned_model.pth",
        "resurrection": True,
        "model_name": MODEL_NAME,
        "max_steps": 50,  # Change to be able to run experiment in a reasonable time
        "head_specific_lr_dict": head_specific_lr,
        "head_specific_patience_dict": head_specific_patience,
        "head_specific_max_epoch_dict": head_specific_max_epoch,
        "logger": Logger(EXPERIMENT_NAME),
    }

    for i in range(30):
        set_random_seed(i)
        try:
            wandb.init(project=EXPERIMENT_NAME, name="run_" + str(i) + "_alltasks")
            trainer = Trainer(task_list=tasks, **config)
            trainer.fit()
            trainer.eval(split=Split.TEST)

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
            print(f"Finetuning with BABE failed with error: {str(e)}")
            raise e

        finally:
            # Always make sure to close wandb
            wandb.finish()
            print("Finetuning with BABE completed successfully!")


if __name__ == "__main__":
    main()