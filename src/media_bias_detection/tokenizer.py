"""Tokenizer module for text preprocessing.

Provides a single tokenizer instance used throughout the model to:
1. Convert text to tokens
2. Maintain consistent vocabulary
3. Prevent duplicate tokenizer creation

The tokenizer is created once and shared across all model components
to save memory and ensure consistent processing.
"""

from transformers import DistilBertTokenizerFast


class Tokenizer:
    """Manages a single shared tokenizer instance.

    Uses DistilBERT's uncased tokenizer as base implementation.
    Created once on first use, then reused for all subsequent calls.
    """

    _instance = None
    _tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Tokenizer, cls).__new__(cls)
            cls._tokenizer = DistilBertTokenizerFast.from_pretrained(
                "distilbert-base-uncased"
            )
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
