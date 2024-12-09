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

from media_bias_detection.data import (
    babe_10,
    cw_hard_03,
    me_too_ma_108,
    good_news_everyone_42
)

from media_bias_detection.training.trainer import Trainer


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    EXPERIMENT_NAME = "experiment_baseline_check"

    MODEL_NAME = "baseline_check"
    selected_tasks = [
        babe_10,
        cw_hard_03,
        me_too_ma_108,
        good_news_everyone_42]

    # Process the data for each task
    for task in selected_tasks:
        for subtask in task.subtasks_list:
            subtask.process()

    # Configure training parameters
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
        "pretrained_path": None,
        "resurrection": True,
        "model_name": MODEL_NAME,
        "head_specific_lr_dict": head_specific_lr,
        "head_specific_patience_dict": head_specific_patience,
        "head_specific_max_epoch_dict": head_specific_max_epoch,
        "logger": Logger(EXPERIMENT_NAME),
    }

    # Set random seed for reproducibility
    set_random_seed()  # default is 321

    try:
        # Initialize wandb
        wandb.init(
            project=EXPERIMENT_NAME,
            name=MODEL_NAME,
            config=config
        )

        # Create trainer
        trainer = Trainer(task_list=selected_tasks, **config)

        # Run debug training
        trainer.fit_debug(k=2)  # Just run one iteration

        # Test evaluation
        trainer.eval(split=Split.TEST)

        # Save the model
        trainer.save_model()

    except Exception as e:
        print(f"Debug training failed with error: {str(e)}")
        raise e

    finally:
        # Always make sure to close wandb
        wandb.finish()

if __name__ == "__main__":
    main()