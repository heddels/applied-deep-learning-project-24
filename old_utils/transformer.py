"""Initialize the tokenizer."""

from transformers import DistilBertTokenizerFast

# Initialize the fast tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')



