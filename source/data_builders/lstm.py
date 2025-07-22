# ==============================================================================
# MODULE: data_builders/lstm.py
# PURPOSE: Builds training data for the LSTM model
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import torch
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
    """Dataset for preparing sequences for a next-character prediction task."""

    def __init__(self, sequences, seq_len, char_to_idx):
        self.seq_len = seq_len
        self.char_to_idx = char_to_idx
        self.idx_to_char = {i: c for c, i in char_to_idx.items()}

        self.X = []
        self.y = []

        for seq in sequences:
            text = seq[1]  # seq is a tuple (id, text)
            if len(text) > self.seq_len:
                for i in range(len(text) - self.seq_len):
                    input_seq = text[i:i + self.seq_len]
                    output_char = text[i + self.seq_len]
                    self.X.append([self.char_to_idx[c] for c in input_seq])
                    self.y.append(self.char_to_idx[output_char])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
