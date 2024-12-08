"""Tokenizer module providing a singleton tokenizer instance."""

from transformers import DistilBertTokenizerFast
from media_bias_detection.utils.logger import general_logger


class Tokenizer:
    """Singleton class to maintain a single tokenizer instance."""

    _instance = None
    _tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Tokenizer, cls).__new__(cls)
            cls._tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        return cls._instance

    def __len__(self):
        """Return vocabulary size"""
        return len(self._tokenizer)  # Add this method

    def __call__(self, *args, **kwargs):
        """Make the tokenizer callable directly."""
        return self._tokenizer(*args, **kwargs)

    @property
    def tokenizer(self):
        return self._tokenizer


# Global tokenizer instance
tokenizer = Tokenizer()