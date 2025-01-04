import os
import sys

import nltk
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from nltk import sent_tokenize
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer

nltk.download('punkt')
tqdm.pandas()
from torch import nn

# Add the repository root to Python path to import your package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class HeadFactory:
    """Decide which head to use for the specific SubTask (st)."""

    @staticmethod
    def create(num_classes):
        return CLSHead(num_classes=num_classes, input_dimension=768, hidden_dimension=768, dropout_prob=0.1)


class CLSHead(nn.Module):
    """Classifier inspired by one used in RoBERTa."""

    def __init__(self, num_classes, input_dimension, hidden_dimension, dropout_prob):
        """Initialize the head."""
        super().__init__()
        self.dense = nn.Linear(input_dimension, hidden_dimension)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.out_proj = nn.Linear(hidden_dimension, num_classes)

    def forward(self, X):
        """Feed the data through head accordingly to RoBERTa approach and compute loss."""
        x = X[:, 0, :]  # take <s> token (equiv. to [CLS])

        # pass CLS through classifier
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        return logits


class BackboneLM(nn.Module):
    """Language encoder model which is shared across all tasks."""

    def __init__(self):
        super(BackboneLM, self).__init__()
        model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.backbone = model


class Model(nn.Module):
    """Main model class."""

    def __init__(self, weight_file, st_id, num_classes):
        """Inititialize model and create heads."""
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.language_model = BackboneLM()
        self.language_model.backbone.pooler = None
        self.st_id = st_id
        self.heads = nn.ModuleDict({st_id: HeadFactory.create(num_classes)})
        self.load_weights(weight_file=weight_file)
        self.to(self.device)
        self.eval()

    def load_weights(self, weight_file):
        pretrained_weights = torch.load(weight_file, map_location=self.device)
        weight_dict = {k: value for k, value in pretrained_weights.items() if "loss" not in k}
        self.load_state_dict(weight_dict, strict=True)

    def forward(self, input_ids, attention_mask):
        x_latent = self.language_model.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        return self.heads[self.st_id](x_latent)


@st.cache_resource
class ModelInference:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = Model(
            weight_file=".streamlit/model_files/finetuned_babe_model_final.pth",
            st_id='10001',
            num_classes=2
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)

    def classify_sentence(self, sent: str):
        toksentence = self.tokenizer(
            sent,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        toksentence = {k: v.to(self.device) for k, v in toksentence.items()}

        self.model.eval()
        with torch.no_grad():
            output = self.model(**toksentence)

        logits = F.softmax(output, dim=1)

        return {
            'unbiased': logits[0][0].item(),
            'biased': logits[0][1].item(),
            'prediction': torch.argmax(logits, dim=1).item()
        }

    def classify_body(self, body):
        sents = sent_tokenize(body)
        results = []
        for sent in sents:
            result = self.classify_sentence(sent)
            results.append({
                'sentence': sent,
                'unbiased_prob': result['unbiased'],
                'biased_prob': result['biased'],
                'prediction': result['prediction']
            })
        return results


# Streamlit interface
st.title("Media Bias Detection")
st.write("This tool analyzes text for potential media bias using a deep learning model.")

# Initialize detector
try:
    detector = ModelInference()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Add before the text input
st.warning("Note: This model works best with English text.")

# Text input
text_input = st.text_area(
    "Enter text to analyze for bias:",
    height=200,
    placeholder="Paste your text here..."
)

# Analysis options
analyze_by_sentence = st.checkbox("Analyze each sentence separately")

if st.button("Analyze"):
    if not text_input:
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            if analyze_by_sentence:
                # Sentence-level analysis
                results = detector.classify_body(text_input)

                # Overall statistics
                bias_scores = [r['biased_prob'] for r in results]
                avg_bias = np.mean(bias_scores)

                # Show overall score
                st.subheader("Overall Analysis")
                st.metric("Average Bias Score", f"{avg_bias:.2%}")

                # Show individual sentences
                st.subheader("Sentence-by-Sentence Analysis")
                for result in results:
                    with st.expander(result['sentence']):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("No Bias", f"{result['unbiased_prob']:.2%}")
                        with col2:
                            st.metric("Bias", f"{result['biased_prob']:.2%}")
            else:
                # Full text analysis
                result = detector.classify_sentence(text_input)

                # Show results
                st.subheader("Analysis Results")
                if result['prediction'] == 1:
                    st.warning("Potential Bias Detected")
                else:
                    st.success("No Significant Bias Detected")

                # Show probabilities
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("No Bias Probability", f"{result['unbiased']:.2%}")
                with col2:
                    st.metric("Bias Probability", f"{result['biased']:.2%}")

# Footer
st.markdown("---")
st.markdown("""
    **Note**: This tool is meant to assist in bias detection but should not be used as the sole determinant. 
    Always apply critical thinking when consuming media.
""")
