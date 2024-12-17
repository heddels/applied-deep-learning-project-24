"""Enums defining key model configurations.

Contains enums for:
1. Dataset splits (train/dev/test/eval)
2. Gradient aggregation methods (mean/PCGrad)
3. Loss scaling approaches (static/uniform)

Used throughout the model to ensure consistent naming and
valid option selection for important parameters.
"""

from enum import Enum


class Split(Enum):
    """Training splits."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    EVAL = "eval"


class AggregationMethod(Enum):
    """Aggregation methods for multi-task learning."""

    PCGRAD = "pcgrad"
    PCGRAD_ONLINE = "pcgrad-online"
    MEAN = "mean"


class LossScaling(Enum):
    """Loss scaling methods."""

    STATIC = "static"
    UNIFORM = "uniform"
