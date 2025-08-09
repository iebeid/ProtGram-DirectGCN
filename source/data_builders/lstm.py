# ==============================================================================
# MODULE: data_builders/lstm.py
# PURPOSE: Builds training data for the LSTM model
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from configuration.config import Config
from source.utils.data import DataUtils, FastaUtils
from source.models.rnn.lstm import LSTM
from source.utils.models import EarlyStopper, EmbeddingProcessor


class LstmPytorchDataset(Dataset):
    """
    A PyTorch Dataset to generate training samples for next-character prediction.
    This replaces the Keras-based LstmCorpusGenerator.
    """

    def __init__(self, text: str, seq_len: int, step: int, char_to_int: Dict[str, int]):
        self.text = text
        self.seq_len = seq_len
        self.step = step
        self.char_to_int = char_to_int
        # The number of sequences is derived from the total text length.
        # The last possible starting index is `len(text) - seq_len - 1`.
        # The number of samples is `floor(last_start_index / step) + 1`.
        # --- FIX: Corrected an off-by-one error in the calculation. ---
        # The original formula was missing a '+1', causing it to miss the last sample.
        self.num_sequences = max(0, (len(self.text) - self.seq_len - 1) // self.step + 1)
        print(f"  [PyTorch Dataset] Corpus has {len(text):,} characters, creating {self.num_sequences:,} samples.")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start_pos = idx * self.step
        input_seq_text = self.text[start_pos: start_pos + self.seq_len]
        target_char = self.text[start_pos + self.seq_len]

        input_seq = torch.tensor([self.char_to_int[c] for c in input_seq_text], dtype=torch.long)
        target = torch.tensor(self.char_to_int[target_char], dtype=torch.long)
        return input_seq, target