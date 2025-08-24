# ==============================================================================
# MODULE: data_builders/lstm.py
# PURPOSE: Builds training data for the LSTM model
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset


class LSTMDataBuilder(Dataset):
    """
    A PyTorch Dataset to generate training samples for next-character prediction.
    This replaces the Keras-based LstmCorpusGenerator.
    """

    def __init__(self, sequences: List[str], seq_len: int, step: int, char_to_int: Dict[str, int]):
        self.sequences = sequences
        self.seq_len = seq_len
        self.step = step
        self.char_to_int = char_to_int
        # --- DEFINITIVE FIX: Generate samples from each sequence individually ---
        self.samples = []
        for seq in self.sequences:
            if len(seq) > self.seq_len:
                for i in range(0, len(seq) - self.seq_len, self.step):
                    input_seq_text = seq[i: i + self.seq_len]
                    target_char = seq[i + self.seq_len]
                    self.samples.append((input_seq_text, target_char))
        print(f"  [PyTorch Dataset] Created {len(self.samples):,} samples from {len(self.sequences)} sequences.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq_text, target_char = self.samples[idx]

        input_seq = torch.tensor([self.char_to_int[c] for c in input_seq_text], dtype=torch.long)
        target = torch.tensor(self.char_to_int[target_char], dtype=torch.long)
        return input_seq, target