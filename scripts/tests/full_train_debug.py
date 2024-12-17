import os
from pathlib import Path
from typing import Dict

import wandb

from media_bias_detection.config.config import head_specific_lr, MAX_NUMBER_OF_STEPS
from media_bias_detection.data import test_tasks
from media_bias_detection.training.trainer import Trainer
from media_bias_detection.training.training_utils import Logger, EarlyStoppingMode
from media_bias_detection.utils.common import set_random_seed
from media_bias_detection.utils.enums import Split, AggregationMethod, LossScaling

os.environ["TOKENIZERS_PARALLELISM"] = "false"

head_specific_patience: Dict[str, int] = {
    "300001": 1,  # CW_HARD task
    "10801": 2,  # MeTooMA task
    "42001": 3,  # GoodNewsEveryone task 1
    "42002": 3  # GoodNewsEveryone task 2
}

# Dictionary mapping subtask IDs to their maximum epochs
head_specific_max_epoch: Dict[str, int] = {
    "300001": 1,  # CW_HARD task
    "10801": 2,  # MeTooMA task
    "42001": 3,  # GoodNewsEveryone task 1
    "42002": 3  # GoodNewsEveryone task 2
}


def main():
    EXPERIMENT_NAME = "debug_task_death"
    MODEL_NAME = "debug_death_model"

    # Add checkpoint path for resuming training_baseline
    CHECKPOINT_PATH = Path("checkpoints") / MODEL_NAME / "latest.pt"
    resume_training = CHECKPOINT_PATH.exists()

    selected_tasks = test_tasks

    # Process the data for each task
    for task in selected_tasks:
        for subtask in task.subtasks_list:
            subtask.process(debug=True)  # Use debug flag

    config = {
        "sub_batch_size": 32,
        "eval_batch_size": 32,
        "initial_lr": 4e-5,
        "dropout_prob": 0.1,
        "hidden_dimension": 768,
        "input_dimension": 768,
        "aggregation_method": AggregationMethod.MEAN,
        "early_stopping_mode": EarlyStoppingMode.HEADS,
        "loss_scaling": LossScaling.STATIC,
        "num_warmup_steps": 2,
        "pretrained_path": None,
        "resurrection": True,
        "head_specific_lr_dict": head_specific_lr,
        "head_specific_patience_dict": head_specific_patience,
        "head_specific_max_epoch_dict": head_specific_max_epoch,
        "model_name": MODEL_NAME,
        "max_steps": MAX_NUMBER_OF_STEPS,
        "logger": Logger(EXPERIMENT_NAME)
    }

    # Set random seed for reproducibility
    set_random_seed()

    try:
        wandb.init(project=EXPERIMENT_NAME, name=MODEL_NAME, config=config)
        trainer = Trainer(task_list=selected_tasks, **config)
        trainer.fit()
        # Test evaluation
        trainer.eval(split=Split.TEST)

        # trainer.save_model()
        print("Debug run completed successfully!")

    except Exception as e:
        print(f"Debug training_baseline failed with error: {str(e)}")
        raise e
    finally:

        wandb.finish()

        # Save the debug model
        trainer.save_model()
        print("Model saved successfully!")


if __name__ == "__main__":
    main()
