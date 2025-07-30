# ==============================================================================
# MODULE: trainers/lstm.py
# PURPOSE: Trainer for a character-level LSTM model to generate protein embeddings.
# VERSION: 8.0 (Refactored to use PyTorch model and training loop)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from configuration.config import Config
from source.utils.data import DataUtils, FastaUtils
from source.models.rnn.lstm import LSTM


class _LstmPytorchDataset(Dataset):
    """
    A PyTorch Dataset to generate training samples for next-character prediction.
    This replaces the Keras-based LstmCorpusGenerator.
    """

    def __init__(self, text: str, seq_len: int, step: int, char_to_int: Dict[str, int]):
        self.text = text
        self.seq_len = seq_len
        self.step = step
        self.char_to_int = char_to_int
        # Calculate the number of sequences that can be generated
        self.num_sequences = (len(self.text) - self.seq_len - 1) // self.step
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


class LSTMBasedEmbedder:
    """
    Trains a character-level LSTM using PyTorch and uses it to generate protein embeddings.
    """

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = self.config.RESULTS_LSTM_EMBEDDINGS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[LSTM] = None
        self.sequences: List[Tuple[str, str]] = []
        self.char_to_int: Dict[str, int] = {}
        self.int_to_char: Dict[int, int] = {}
        self.vocab_size = 0
        print(f"LSTMBasedEmbedder initialized. Using device: {self.device}")

    def _prepare_corpus(self):
        """
        Prepares the corpus by loading sequences and applying LSTM-specific downsampling.
        This method is framework-agnostic and remains unchanged.
        """
        print("  Preparing LSTM corpus and character mappings...")
        all_loaded_sequences = list(FastaUtils.parse_sequences(self.config.SEQUENCE_FILE_PATHS))

        should_downsample = self.config.LSTM_DOWNSAMPLE_FRACTION and 0 < self.config.LSTM_DOWNSAMPLE_FRACTION < 1.0
        if should_downsample:
            sample_size = int(len(all_loaded_sequences) * self.config.LSTM_DOWNSAMPLE_FRACTION)
            print(f"  Applying LSTM-specific downsampling: {self.config.LSTM_DOWNSAMPLE_FRACTION:.1%}")
            print(f"    - Sampling {sample_size} of {len(all_loaded_sequences)} sequences.")
            self.sequences = random.sample(all_loaded_sequences, sample_size)
        else:
            self.sequences = all_loaded_sequences

        if not self.sequences:
            print("  WARNING: No sequences available for LSTM training after downsampling. Skipping.")
            self.vocab_size = 0
            return

        all_chars = sorted(list(set("".join(seq for _, seq in self.sequences))))
        self.char_to_int = {c: i for i, c in enumerate(all_chars)}
        self.int_to_char = {i: c for i, c in enumerate(all_chars)}
        self.vocab_size = len(all_chars)
        print(f"  Vocabulary size: {self.vocab_size}")

    def _build_model(self):
        """Builds the PyTorch LSTM model for next-character prediction."""
        self.model = LSTM(
            vocab_size=self.vocab_size,
            embedding_dim=self.config.LSTM_EMBEDDING_DIM,
            hidden_dim=self.config.LSTM_HIDDEN_DIM,
            num_layers=self.config.LSTM_NUM_LAYERS
        ).to(self.device)
        print("  PyTorch LSTM model built:")
        print(self.model)

    def run(self) -> Optional[str]:
        DataUtils.print_header("PIPELINE: Training LSTM & Generating Embeddings (PyTorch)")
        self._prepare_corpus()
        if not self.sequences or self.vocab_size == 0:
            print("  Aborting LSTM pipeline due to lack of data.")
            return None

        self._build_model()
        assert self.model is not None, "Model must be built before running."

        # --- PyTorch Training Loop ---
        print(f"  Training LSTM model for {self.config.LSTM_EPOCHS} epochs...")
        corpus_text = "".join([seq for _, seq in self.sequences])
        dataset = _LstmPytorchDataset(
            text=corpus_text,
            seq_len=self.config.LSTM_TRAIN_SEQ_LEN,
            step=self.config.LSTM_TRAIN_STEP,
            char_to_int=self.char_to_int
        )
        dataloader = DataLoader(dataset, batch_size=self.config.LSTM_BATCH_SIZE, shuffle=True, num_workers=4)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LSTM_LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.config.LSTM_EPOCHS):
            epoch_loss = 0
            for inputs, targets in tqdm(dataloader, desc=f"  Epoch {epoch + 1}/{self.config.LSTM_EPOCHS}", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            print(f"  Epoch {epoch + 1} finished. Average Loss: {avg_loss:.4f}")

        # --- PyTorch Inference for Embeddings ---
        print("\n  LSTM training complete. Generating embeddings...")
        self.model.eval()
        protein_embeddings = {}
        batch_size = self.config.LSTM_BATCH_SIZE

        print("  Sorting sequences by length for efficient batching...")
        sorted_sequences = sorted(self.sequences, key=lambda x: len(x[1]))

        with torch.no_grad():
            for i in tqdm(range(0, len(sorted_sequences), batch_size), desc="  Generating Embeddings in Batches"):
                batch = sorted_sequences[i:i + batch_size]
                if not batch: continue

                batch_ids = [item[0] for item in batch]
                batch_seqs_text = [item[1] for item in batch]

                tokenized_batch = []
                original_lengths = []
                for seq_text in batch_seqs_text:
                    tokens = [self.char_to_int.get(c) for c in seq_text]
                    tokens = [t for t in tokens if t is not None]  # Filter out unknown characters
                    if tokens:
                        tokenized_batch.append(torch.tensor(tokens, dtype=torch.long))
                        original_lengths.append(len(tokens))

                if not tokenized_batch: continue

                # Pad sequences for batch processing
                padded_batch = pad_sequence(tokenized_batch, batch_first=True, padding_value=0).to(self.device)

                # Get hidden states for all time steps
                embedded = self.model.embedding(padded_batch)
                all_hidden_states_batch, _ = self.model.lstm(embedded)
                all_hidden_states_batch = all_hidden_states_batch.cpu().numpy()

                for j in range(len(all_hidden_states_batch)):
                    original_len = original_lengths[j]
                    valid_hidden_states = all_hidden_states_batch[j, :original_len, :]
                    pooled_embedding = np.mean(valid_hidden_states, axis=0)
                    protein_embeddings[batch_ids[j]] = pooled_embedding

        output_path = self.config.RESULTS_LSTM_EMBEDDINGS_DIR / "lstm_generated_embeddings.h5"
        DataUtils.write_h5(protein_embeddings, output_path, "Writing LSTM Embeddings")
        print(f"\nSUCCESS: LSTM embeddings saved to: {output_path}")
        return str(output_path)
