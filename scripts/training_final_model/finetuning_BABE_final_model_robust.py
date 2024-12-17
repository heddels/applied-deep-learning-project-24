"""Script for executing robust evaluation of BABE finetuning with optimal hyperparameters."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from sklearn.metrics import confusion_matrix

from media_bias_detection.config.config import (
    head_specific_lr,
    head_specific_max_epoch,
    head_specific_patience,
    head_specific_sub_batch_size,
)
from media_bias_detection.data import st_1_babe_10 as babe
from media_bias_detection.data.task import Task
from media_bias_detection.training.trainer import Trainer
from media_bias_detection.training.training_utils import Logger, EarlyStoppingMode
from media_bias_detection.utils.common import set_random_seed
from media_bias_detection.utils.enums import Split, AggregationMethod, LossScaling

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def plot_learning_curves(run_history, results_dir):
    """Plot learning curves from a training run."""
    metrics = ['loss', 'accuracy', 'f1_score']
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15))

    for idx, metric in enumerate(metrics):
        # Plot training metric
        train_metric = [h[f'train_{metric}'] for h in run_history if f'train_{metric}' in h]
        steps = range(len(train_metric))
        axes[idx].plot(steps, train_metric, label='Training')

        # Plot validation metric
        val_metric = [h[f'dev_{metric}'] for h in run_history if f'dev_{metric}' in h]
        if val_metric:
            val_steps = range(0, len(steps), 3)  # Since we validate every 3 steps
            axes[idx].plot(val_steps, val_metric, label='Validation')

        axes[idx].set_title(f'{metric} over Training Steps')
        axes[idx].set_xlabel('Steps')
        axes[idx].set_ylabel(metric)
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(results_dir / "learning_curves.png")
    plt.close()


def create_confusion_matrix(trainer, split, results_dir):
    """Create and plot confusion matrix for the model."""
    # Get predictions and true labels from the test set
    y_true = []
    y_pred = []

    # Use the test dataloader
    dataloader = trainer.batch_lists[split].dataloaders['10001']  # BABE task ID

    trainer.model.eval()
    with torch.no_grad():
        for batch in dataloader:
            X, attention_masks, Y, _ = batch
            X = X.to(trainer.model.device)
            attention_masks = attention_masks.to(trainer.model.device)

            # Get model predictions
            outputs = trainer.model.language_model.backbone(
                input_ids=X,
                attention_mask=attention_masks
            ).last_hidden_state

            logits = trainer.model.heads['10001'](outputs, None)[0]
            predictions = torch.argmax(logits, dim=1)

            y_true.extend(Y.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Biased', 'Biased'],
                yticklabels=['Not Biased', 'Biased'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(results_dir / "confusion_matrix.png")
    plt.close()

    return cm


def save_results(results, experiment_name, run_histories=None):
    """Save results to CSV and generate plots"""
    # Create results directory if it doesn't exist
    results_dir = Path("results") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Save raw results
    df.to_csv(results_dir / "raw_results.csv", index=False)

    # Calculate and save summary statistics
    summary = df.agg(['mean', 'std', 'min', 'max'])
    summary.to_csv(results_dir / "summary_statistics.csv")

    # Create box plots
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['accuracy', 'f1_score']])
    plt.title('Distribution of Metrics Across Runs')
    plt.savefig(results_dir / "metrics_distribution.png")
    plt.close()

    # Print summary
    print("\nResults Summary:")
    print(f"Accuracy: {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
    print(f"F1 Score: {df['f1_score'].mean():.4f} ± {df['f1_score'].std():.4f}")

    # Add learning curves if histories are provided
    if run_histories:
        # Plot average learning curves across all runs
        all_histories = pd.DataFrame(run_histories)
        plot_learning_curves(all_histories, results_dir)


def main():
    EXPERIMENT_NAME = "babe_robust_evaluation_final"
    MODEL_NAME = "finetuned_babe_model_final"

    results = []
    run_histories = []
    confusion_matrices = []

    tasks = [Task(task_id=babe.id, subtasks_list=[babe])]

    for t in tasks:
        for st in t.subtasks_list:
            st.process()

    # Use optimal hyperparameters found
    config = {
        "head_specific_sub_batch_size": head_specific_sub_batch_size,  # Optimal from hyperparameter search
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
            print(f"\nStarting run {i + 1}/30 with seed {seed}")

            run = wandb.init(
                project=EXPERIMENT_NAME,
                name=f"run_{i}_seed_{seed}",
                config=config,
                reinit=True
            )

            trainer = Trainer(task_list=tasks, **config)
            trainer.fit()

            # Evaluate on test set
            trainer.eval(split=Split.TEST)

            # Get metrics for this run
            metrics = {
                'seed': seed,
                'accuracy': trainer.tracker.metrics[Split.TEST]['10001']['acc'].mean_all(),
                'f1_score': trainer.tracker.metrics[Split.TEST]['10001']['f1'].mean_all(),
            }

            # Store run history for learning curves
            run_histories.append(run.history())

            # Create confusion matrix
            cm = create_confusion_matrix(trainer, Split.TEST, Path("results") / EXPERIMENT_NAME)
            confusion_matrices.append(cm)

            results.append(metrics)
            print(f"Run {i + 1} completed: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1_score']:.4f}")

        except Exception as e:
            print(f"Run {i + 1} failed with error: {str(e)}")
            continue

        finally:
            wandb.finish()

    # Save and analyze results
    save_results(results, EXPERIMENT_NAME, run_histories)

    # Calculate and save average confusion matrix
    avg_cm = np.mean(confusion_matrices, axis=0)
    std_cm = np.std(confusion_matrices, axis=0)

    # Plot average confusion matrix with standard deviation
    plt.figure(figsize=(10, 7))
    sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Not Biased', 'Biased'],
                yticklabels=['Not Biased', 'Biased'])

    # Add standard deviation information in the annotations
    for i in range(avg_cm.shape[0]):
        for j in range(avg_cm.shape[1]):
            plt.text(j + 0.5, i + 0.7,
                     f'±{std_cm[i, j]:.2f}',
                     ha='center', va='center',
                     color='black' if avg_cm[i, j] < avg_cm.max() / 2 else 'white')

    plt.title('Average Confusion Matrix (30 runs)\nMean ± Standard Deviation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(Path("results") / EXPERIMENT_NAME / "average_confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    main()
