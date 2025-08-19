import traceback
from typing import TYPE_CHECKING
import argparse
import sys
from pathlib import Path

# --- FIX: Add the project root to the Python path to allow relative imports ---
# This must be done BEFORE any local modules (like 'configuration') are imported.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from configuration.config import Config
# --- Local import to avoid circular dependency at module level ---
from transformers import AutoTokenizer, TFAutoModel

if TYPE_CHECKING:
    pass

class EarlyStopper:
    """A simple early stopper to monitor loss and stop training when it stops improving."""

    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False