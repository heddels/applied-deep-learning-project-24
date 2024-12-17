"""Backbone model module providing the shared language model."""

import torch
from transformers import DistilBertModel

from media_bias_detection.training.gradient import GradsWrapper
from media_bias_detection.utils.logger import general_logger


class BackboneLM(GradsWrapper):
    """Language model backbone shared across all tasks.

    This class wraps the pretrained DistilBERT model and handles
    gradient manipulation for the shared parameters.

    Attributes:
        backbone: The underlying DistilBERT model
    """

    def __init__(self, pretrained_path: str = None):
        """Initialize the backbone model.

        Args:
            pretrained_path: Optional path to pretrained weights
        """
        super().__init__()

        try:
            general_logger.info("Initializing backbone language model")
            self.backbone = DistilBertModel.from_pretrained('distilbert-base-uncased')

            if pretrained_path:
                self.load_pretrained(pretrained_path)

        except Exception as e:
            general_logger.error(f"Failed to initialize backbone: {str(e)}")
            raise

    def load_pretrained(self, path: str) -> None:
        """Load pretrained weights.

        Args:
            path: Path to pretrained weights

        Raises:
            RuntimeError: If loading fails
        """
        try:
            state_dict = torch.load(path)
            self.backbone.load_state_dict(state_dict)
            general_logger.info(f"Loaded pretrained weights from {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained weights: {str(e)}")
