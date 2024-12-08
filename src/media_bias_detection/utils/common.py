import os
from box.exceptions import BoxValueError
import yaml
from media_bias_detection.utils.logger import general_logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import functools
import random

import numpy as np
import torch

from media_bias_detection.config.config import RANDOM_SEED


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            general_logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            general_logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    general_logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    general_logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    general_logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    general_logger.info(f"binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

#from here one directly taken from magpie repo utils
"""This module contains utils."""
@ensure_annotations
def integer_formatter(i):
    """Format integers.

    Example:
        1827028 -> 1,827,028
    """
    if isinstance(i, str):
        return i
    return f"{i:,d}"

@ensure_annotations
def float_formatter(i):
    """Format floats.

    Example:
        0.8982232 -> 0.898
    """
    if isinstance(i, str):
        return i
    return f"{i:0.3f}"

@ensure_annotations
def get_class_weights(y, method="ins"):
    """Compute the weights for vector of counts of each label.

    ins = inverse number of samples
    isns = inverse squared number of samples
    esns = effective sampling number of samples
    """
    counts = y.unique(return_counts=True)[1]

    if method == "ins":
        weights = 1.0 / counts
        weights = weights / sum(weights)
    if method == "isns":
        weights = 1.0 / torch.pow(counts, 0.5)
        weights = weights / sum(weights)
    if method == "esns":
        beta = 0.999
        weights = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
        weights = weights / sum(weights)

    return weights

@ensure_annotations
def set_random_seed(seed=RANDOM_SEED):
    """Random seed for comparable results."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.cuda.manual_seed_all(seed)

@ensure_annotations
def rsetattr(obj, attr, val):
    """Set an attribute recursively.

    Inspired by
    https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties.
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

@ensure_annotations
def rgetattr(obj, attr, *args):
    """Get an attribute recursively.

    Inspired by
    https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))