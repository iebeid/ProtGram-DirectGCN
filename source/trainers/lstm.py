# ==============================================================================
# MODULE: trainers/lstm.py
# PURPOSE: Trainer for a character-level LSTM model to generate protein embeddings.
# VERSION: 9.0 (Integrated consistent ID mapping)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

from typing import List, Tuple, Dict, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from pathlib import Path
from configuration.config import Config
from source.utils.data.data_utils import DataUtils
from source.utils.data.fasta_utils import FastaUtils
from source.utils.data.id_mapper import IDMapper
from source.utils.fs.file_utils import FileUtils
from source.models.rnn.lstm import LSTM
from source.data_builders.lstm import LSTMDataBuilder
from source.utils.models.early_stopper import EarlyStopper


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
        Prepares the corpus by loading sequences. Downsampling is now handled
        globally by the main pipeline runner.
        """
        self.sequences = list(FastaUtils.parse_sequences(
            self.config.SEQUENCE_FILE_PATHS,
            perform_cleaning=self.config.PROTGRAM_CLEAN_FASTA_ON_PARSE,
            min_len=self.config.PROTGRAM_FASTA_MIN_LEN,
            max_len=self.config.PROTGRAM_FASTA_MAX_LEN,
            alphabet_type=self.config.PROTGRAM_FASTA_ALPHABET
        ))

        if not self.sequences:
            print("  WARNING: No sequences available for LSTM training. Skipping.")
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
            num_layers=self.config.LSTM_NUM_LAYERS,
            dropout_rate=self.config.LSTM_DROPOUT_RATE
        ).to(self.device)
        print("  PyTorch LSTM model built:")
        print(self.model)

    def _train_model(self):
        """
        Handles the training loop for the LSTM model, including validation and early stopping.
        """
        assert self.model is not None, "Model must be built before training."

        # More memory efficient: create a single text corpus and split it.
        # FIX: Split the list of sequences, not a list containing one giant string,
        # to prevent a ValueError from train_test_split when n_samples=1.
        train_seq_data, val_seq_data = train_test_split(
            self.sequences, test_size=self.config.LSTM_VALIDATION_SPLIT, random_state=self.config.RANDOM_STATE
        )

        train_dataset = LSTMDataBuilder(
            sequences=[seq for _, seq in train_seq_data],
            seq_len=self.config.LSTM_TRAIN_SEQ_LEN,
            step=self.config.LSTM_TRAIN_STEP,
            char_to_int=self.char_to_int
        )
        val_dataset = LSTMDataBuilder(
            sequences=[seq for _, seq in val_seq_data],
            seq_len=self.config.LSTM_TRAIN_SEQ_LEN,
            step=self.config.LSTM_TRAIN_STEP,
            char_to_int=self.char_to_int
        )

        num_workers = getattr(self.config, 'DATALOADER_WORKERS', 0)
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.LSTM_BATCH_SIZE, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=self.config.LSTM_BATCH_SIZE, shuffle=False, num_workers=num_workers)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LSTM_LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        early_stopper = EarlyStopper(
            patience=self.config.LSTM_EARLY_STOPPING_PATIENCE,
            min_delta=self.config.LSTM_EARLY_STOPPING_MIN_DELTA
        )

        for epoch in range(self.config.LSTM_EPOCHS):
            self.model.train()
            epoch_loss = 0
            for inputs, targets in tqdm(train_dataloader, desc=f"  Epoch {epoch + 1}/{self.config.LSTM_EPOCHS}", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                # --- FIX: Unpack the tuple returned by the model ---
                logits, _ = self.model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_train_loss = epoch_loss / len(train_dataloader)

            # Validation loop
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    # --- FIX: Unpack the tuple returned by the model ---
                    logits, _ = self.model(inputs)
                    loss = criterion(logits, targets)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"  Epoch {epoch + 1} finished. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if self.config.LSTM_USE_EARLY_STOPPING:
                if early_stopper.early_stop(avg_val_loss):
                    print(f"  Early stopping triggered at epoch {epoch + 1}.")
                    break

    def run(self) -> Optional[Dict[str, str]]:
        mlflow.set_experiment(self.config.MLFLOW_LLMS_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"LSTM_{Path(self.config.SEQUENCE_FILE_PATHS[0]).stem}"):
            mlflow.log_params(self.config.get_as_dict('lstm'))
            DataUtils.print_header("PIPELINE: Training LSTM & Generating Embeddings (PyTorch)")
            self._prepare_corpus()
            if not self.sequences or self.vocab_size == 0:
                print("  Aborting LSTM pipeline due to lack of data.")
                return None

            self._build_model()
            self._train_model()

            print("\n  LSTM training complete. Generating embeddings...")
            assert self.model is not None, "Model must be trained before inference."
            self.model.eval()
            protein_embeddings = {}
            batch_size = self.config.LSTM_BATCH_SIZE

            sorted_sequences = sorted(self.sequences, key=lambda x: len(x[1]))

            with torch.no_grad():
                for i in tqdm(range(0, len(sorted_sequences), batch_size), desc="  Generating Embeddings in Batches"):
                    batch = sorted_sequences[i:i + batch_size]
                    if not batch: continue

                    batch_ids = [item[0] for item in batch]
                    batch_seqs_text = [item[1] for item in batch]

                    tokenized_batch = [
                        torch.tensor([self.char_to_int[c] for c in seq if c in self.char_to_int], dtype=torch.long)
                        for seq in batch_seqs_text
                    ]
                    valid_indices = [i for i, t in enumerate(tokenized_batch) if len(t) > 0]
                    if not valid_indices: continue

                    tokenized_batch = [tokenized_batch[i] for i in valid_indices]
                    batch_ids = [batch_ids[i] for i in valid_indices]

                    padded_batch = pad_sequence(tokenized_batch, batch_first=True, padding_value=0).to(self.device)

                    _, batch_embeddings = self.model(padded_batch)
                    for j, prot_id in enumerate(batch_ids):
                        protein_embeddings[prot_id] = batch_embeddings[j].cpu().numpy().astype(np.float16)

            output_path = self.config.RESULTS_LSTM_EMBEDDINGS_DIR / "lstm_generated_embeddings.h5"
            FileUtils.write_h5(protein_embeddings, output_path, "Writing LSTM Embeddings")

            print(f"\nSUCCESS: LSTM embeddings saved to: {output_path}")
            # --- DEFINITIVE FIX: Return a dictionary for consistency with other trainers ---
            output_paths = {"LSTM-Generated": str(output_path)}
            return output_paths