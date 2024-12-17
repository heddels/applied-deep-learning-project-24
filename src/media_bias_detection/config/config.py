"""Configuration management for the Media Bias Detection MTL model.

This module centralizes all configuration parameters for the multi-task learning model,
including:
- Dataset mappings and task family definitions
- Training hyperparameters
- Model architecture settings
- Task-specific configurations

The configuration is organized into several key sections:
1. Dataset Mappings: Links dataset IDs to names and task families
2. Global Training Parameters: Steps, seeds, and data splits
3. Hyperparameter Search Spaces: Used for optimization
4. Task-Specific Parameters: Learning rates, patience values, and epochs per task

Note: Task IDs in dictionaries follow the format:
    - First 2-3 digits: Dataset ID (e.g., 10 for BABE)
    - Last 2 digits: Subtask number (e.g., 01 for first subtask)
    Example: "10001" = BABE dataset (10), first subtask (01)
"""

import os
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")


dataset_id_to_dataset_name = {
    3: "CW_HARD",
    10: "BABE",
    12: "PHEME",
    22: "NewsWCL50",
    42: "GoodNewsEveryone",
    103: "MPQA",
    108: "MeTooMA",
    109: "Stereotype",
    116: "MDGender",
    128: "GWSD",
}


class TaskFamilies(Enum):
    """Task Families."""

    SUBJECTIVITY = "Subjectivity"
    MEDIA_BIAS = "Media Bias"
    HATE_SPEECH = "Hate Speech"
    GENDER_BIAS = "Gender Bias"
    SENTIMENT_ANALYSIS = "Sentiment Analysis"
    FAKE_NEWS_DETECTION = "Fake News Detection"
    GROUP_BIAS = "Group Bias"
    EMOTIONALITY = "Emotionality"
    STANCE_DETECTION = "Stance Detection"


dataset_id_to_family = {
    3: TaskFamilies.SUBJECTIVITY,  # new dataset the old for category subjectivity wrongly assigned there
    10: TaskFamilies.MEDIA_BIAS,
    22: TaskFamilies.SUBJECTIVITY,
    42: TaskFamilies.EMOTIONALITY,
    103: TaskFamilies.SENTIMENT_ANALYSIS,
    108: TaskFamilies.HATE_SPEECH,
    # original code had gender bias as task family but from dataset description it is hate speech
    109: TaskFamilies.GROUP_BIAS,
    116: TaskFamilies.GENDER_BIAS,
    128: TaskFamilies.STANCE_DETECTION,
}

MAX_NUMBER_OF_STEPS = 200  # changed from 1000 to 100 for pre-finetuning and 50 for finetuning for baseline

# Task-configs
MAX_LENGTH = 128

RANDOM_SEED = 321

TRAIN_RATIO, DEV_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

# hyperparameter ranges for hyperparameter tuning
hyper_param_dict = {
    "dropout_prob": {"values": [0.1, 0.2, 0.3]},
    "sub_batch_size": {"values": [16, 32, 64]},
    "num_warmup_steps": {"values": [50, 100]},
}

# Head specific configurations
head_specific_lr = {
    "300001": 0.0001,
    "10001": 3e-05,
    "10002": 2e-05,
    "10301": 0.0001,
    "10801": 2e-05,
    "10901": 4e-05,
    "10902": 2e-05,
    "11601": 4e-05,
    "42001": 2e-05,
    "42002": 2e-05,
    "12001": 2e-05,
    "12002": 4e-05,
    "12801": 0.0001,
}

head_specific_patience = {
    "300001": 50,
    "10001": 75,
    "10002": 100,
    "10301": 75,
    "10801": 25,
    "10901": 75,
    "10902": 75,
    "11601": 75,
    "42001": 100,
    "42002": 100,
    "12001": 100,
    "12002": 100,
    "12801": 75,
}

head_specific_max_epoch = {
    "300001": 3,
    "10001": 3,
    "10002": 3,
    "10301": 3,
    "10801": 3,
    "10901": 3,
    "10902": 5,
    "11601": 10,
    "42001": 3,
    "42002": 10,
    "12001": 3,
    "12002": 3,
    "12801": 3,
}
