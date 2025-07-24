# ==============================================================================
# MODULE: data_builders/lstm.py
# PURPOSE: Builds training data for the LSTM model
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

from typing import List, Tuple

import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical


# LstmCorpusGenerator class remains the same and is correct.
class LstmCorpusGenerator(Sequence):
    """
    Generates batches of data for the LSTM model on-the-fly.
    This version uses a step parameter to control the sliding window, allowing
    for much faster, non-overlapping sequence generation.
    """

    def __init__(self, sequences: List[Tuple[str, str]], batch_size: int, seq_len: int, step: int, vocab_size: int,
                 char_to_int: dict):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.step = step
        self.vocab_size = vocab_size
        self.char_to_int = char_to_int

        print("  [Generator] Concatenating sequences for corpus...")
        self.text = "".join([seq for _, seq in sequences])
        self.text_len = len(self.text)
        print(f"  [Generator] Corpus created with {self.text_len:,} characters.")

        self.num_sequences = (self.text_len - self.seq_len) // self.step
        print(f"  [Generator] Created {self.num_sequences:,} training samples with a step of {self.step}.")

    def __len__(self):
        return self.num_sequences // self.batch_size

    def __getitem__(self, index):
        batch_x, batch_y = [], []
        start_sequence_index = index * self.batch_size

        for i in range(self.batch_size):
            current_sequence_index = start_sequence_index + i
            text_start_pos = current_sequence_index * self.step
            text_end_pos = text_start_pos + self.seq_len

            if text_end_pos >= self.text_len:
                continue

            input_seq = self.text[text_start_pos:text_end_pos]
            output_char = self.text[text_end_pos]

            batch_x.append([self.char_to_int[char] for char in input_seq])
            batch_y.append(self.char_to_int[output_char])

        return np.array(batch_x), to_categorical(batch_y, num_classes=self.vocab_size)
