import os
import torch
from pathlib import Path

class ModelCheckpoint:
    def __init__(
            self,
            save_dir='checkpoints',
            save_freq=50,
            monitor='loss',
            verbose=True
    ):
        """Simple checkpoint manager."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = save_freq
        self.monitor = monitor
        self.verbose = verbose

    def save_checkpoint(self, model, step, metrics):
        """Save model checkpoint."""
        filepath = self.save_dir / f'checkpoint_step_{step}.pth'
        torch.save(model.state_dict(), filepath)

        if self.verbose:
            print(f'Saved checkpoint to {filepath}')

    def load_checkpoint(self, filepath, model):
        """Load model checkpoint."""
        state_dict = torch.load(filepath)
        model.load_state_dict(state_dict)
        if self.verbose:
            print(f'Loaded checkpoint from {filepath}')