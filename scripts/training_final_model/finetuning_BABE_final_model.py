"""Final evaluation script for BABE task finetuning for saving the final model.

Performs robust evaluation by:
1. Running on best-performing random seed
2. Using optimal hyperparameters from search

Results are logged to wandb for analysis.
"""

import os

import wandb

from media_bias_detection.config.config import (
    head_specific_lr,
    head_specific_max_epoch,
    head_specific_patience,
)
from media_bias_detection.data import st_1_babe_10 as babe
from media_bias_detection.data.task import Task
from media_bias_detection.training.trainer import Trainer
from media_bias_detection.training.training_utils import Logger, EarlyStoppingMode
from media_bias_detection.utils.common import set_random_seed
from media_bias_detection.utils.enums import Split, AggregationMethod, LossScaling

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    EXPERIMENT_NAME = "babe_evaluation_final"
    MODEL_NAME = "finetuned_babe_model_final"

    results = []

    tasks = [Task(task_id=babe.id, subtasks_list=[babe])]

    for t in tasks:
        for st in t.subtasks_list:
            st.process()

    # Use optimal hyperparameters found
    config = {
        "sub_batch_size": 64,  # Optimal from hyperparameter search
        "eval_batch_size": 128,
        "initial_lr": 4e-5,
        "dropout_prob": 0.1,  # Optimal from hyperparameter search
        "hidden_dimension": 768,
        "input_dimension": 768,
        "aggregation_method": AggregationMethod.MEAN,
        "early_stopping_mode": EarlyStoppingMode.HEADS,
        "loss_scaling": LossScaling.STATIC,
        "num_warmup_steps": 100,  # Optimal from hyperparameter search
        "pretrained_path": "model_files/pre_finetuned_model.pth",
        "resurrection": True,
        "model_name": MODEL_NAME,
        "max_steps": 500,
        "head_specific_lr_dict": head_specific_lr,
        "head_specific_patience_dict": head_specific_patience,
        "head_specific_max_epoch_dict": head_specific_max_epoch,
        "logger": Logger(EXPERIMENT_NAME),
    }

    # Run multiple evaluations with different seeds

    seed = 8
    set_random_seed(seed)

    try:
        print(f"\nStarting run with seed {seed}")

        wandb.init(
            project=EXPERIMENT_NAME,
            name="final_model",
        )

        trainer = Trainer(task_list=tasks, **config)
        trainer.fit()
        trainer.eval(split=Split.TEST)

        metrics = {
            "seed": seed,
            "accuracy": trainer.tracker.metrics[Split.TEST]["10001"][
                "acc"
            ].mean_all(),
            "f1_score": trainer.tracker.metrics[Split.TEST]["10001"][
                "f1"
            ].mean_all(),
            "loss": trainer.tracker.losses[Split.TEST]["10001"].mean_all(),
        }
        print(
            f"Run completed: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}, Loss = {metrics['loss']:.4f}"
        )
        trainer.save_model()
        print("Model saved")

    except Exception as e:
        print(f"Run failed with error: {str(e)}")

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
