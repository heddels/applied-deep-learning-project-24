"""Main MTL model implementation."""

from typing import Dict, Tuple, Optional
import torch
from torch import nn
from typing import List

from ..tokenizer import tokenizer
from .backbone import BackboneLM
from .heads import HeadFactory
from ..utils.logger import general_logger
from ..data.dataset import BatchData


class Model(nn.Module):
    """MTL model combining backbone and task-specific heads."""

    def __init__(self, stl: List, *args, **kwargs):
        """Initialize model with subtasks list and create task-specific heads.

        Args:
            stl: List of subtasks to create heads for
            *args: Additional positional arguments for heads
            **kwargs: Additional keyword arguments for heads
        """
        super().__init__()
        self.stl = stl
        self.subtask_id_to_subtask = {int(f"{st.id}"): st for st in stl}
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

        # Initialize backbone
        self.language_model = BackboneLM()
        self.language_model.backbone.resize_token_embeddings(len(tokenizer))
        # Initialize heads
        self.heads = nn.ModuleDict({str(st.id): HeadFactory(st, *args, **kwargs) for st in stl})

        # Move model to device
        self.to(self.device)
        general_logger.info(f"Initialized model with {len(self.heads)} heads on {self.device}")

    def forward(self, X, attention_masks, Y, st_id):
        """Pass data through model and task-specific head.

        Args:
            X: Input tensor
            attention_masks: Attention mask tensor
            Y: Target tensor
            st_id: Subtask ID

        Returns:
            Tuple of (loss, metrics)
        """
        # Pass through backbone
        with torch.set_grad_enabled(self.training):
            x_enc = self.language_model.backbone(
                input_ids=X,
                attention_mask=attention_masks
            ).last_hidden_state

            # Pass through task-specific head
            head = self.heads[str(st_id.item())]
            logits, loss, metrics = head(x_enc, Y)

            return loss, metrics