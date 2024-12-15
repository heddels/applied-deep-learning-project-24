#enum file from original repo combined into one skript
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
