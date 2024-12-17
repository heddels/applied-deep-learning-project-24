"""Final evaluation script for BABE task finetuning.

Performs robust evaluation by:
1. Running 10 trials with different random seeds
2. Using optimal hyperparameters from search
3. Tracking accuracy, F1-score and loss
4. Saving detailed statistics and summaries

Results are saved to results/babe_robust_evaluation_final/
and logged to wandb for analysis.
"""

import os
from pathlib import Path

import pandas as pd
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


def save_results(results, experiment_name):
    """Save results to CSV and generate plots"""
    results_dir = Path("results") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    df.to_csv(results_dir / "raw_results.csv", index=False)

    summary = pd.DataFrame(
        {
            "Metric": ["Accuracy", "F1 Score", "Loss"],
            "Mean": [df["accuracy"].mean(), df["f1_score"].mean(), df["loss"].mean()],
            "Std Dev": [df["accuracy"].std(), df["f1_score"].std(), df["loss"].std()],
            "Min": [df["accuracy"].min(), df["f1_score"].min(), df["loss"].min()],
            "Max": [df["accuracy"].max(), df["f1_score"].max(), df["loss"].max()],
        }
    )

    summary.to_csv(results_dir / "summary_results.csv", index=False)

    print("\nResults Summary:")
    print(summary.to_string(float_format=lambda x: "{:.4f}".format(x)))


def main():
    EXPERIMENT_NAME = "babe_robust_evaluation_final"
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
    for i in range(10):
        seed = i
        set_random_seed(seed)

        try:
            print(f"\nStarting run {i + 1}/10 with seed {seed}")

            run = wandb.init(
                project=EXPERIMENT_NAME,
                name=f"run_{i}_seed_{seed}",
                config=config,
                reinit=True,
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
            results.append(metrics)
            print(
                f"Run {i + 1} completed: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}, Loss = {metrics['loss']:.4f}"
            )

        except Exception as e:
            print(f"Run {i + 1} failed with error: {str(e)}")
            continue

        finally:
            wandb.finish()

    save_results(results, EXPERIMENT_NAME)


if __name__ == "__main__":
    main()
