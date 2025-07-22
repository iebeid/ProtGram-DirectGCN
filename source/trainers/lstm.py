# ==============================================================================
# MODULE: trainers/lstm.py
# PURPOSE: Trainer for a character-level LSTM model to generate protein embeddings.
# VERSION: 2.0 (Corrected output path and used configured sequence length)
# AUTHOR: Islam Ebeid
# ==============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configuration.config import Config
from source.data_builders.lstm import LSTMDataset
from source.models.ml.lstm import LSTM
from source.utils.data import DataUtils, FastaUtils


class LSTMBasedEmbedder:
    """
    Trains a character-level LSTM on a next-character prediction task and then
    uses the trained model to generate a single embedding vector for each
    protein sequence.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self) -> str:
        """
        Main function to train the LSTM and generate embeddings.

        Returns:
            str: The file path to the generated HDF5 embedding file.
        """
        DataUtils.print_header("PIPELINE: Training LSTM & Generating Embeddings")

        # 1. Load data and create vocabulary
        sequences = list(FastaUtils.parse_sequences(self.config.SEQUENCE_FILE_PATHS))
        if not sequences:
            print("  No sequences found in FASTA files. Skipping LSTM pipeline.")
            return ""

        chars = sorted(list(set("".join(s[1] for s in sequences))))
        char_to_idx = {c: i for i, c in enumerate(chars)}
        vocab_size = len(chars)

        # 2. Create Dataset and DataLoader
        # Use the sequence length defined in the configuration
        train_seq_len = self.config.LSTM_TRAIN_SEQ_LEN
        dataset = LSTMDataset(sequences, train_seq_len, char_to_idx)
        if not dataset:
            print("  No training data generated for LSTM. Sequence lengths might be too short. Skipping.")
            return ""
        dataloader = DataLoader(dataset, batch_size=self.config.LSTM_BATCH_SIZE, shuffle=True)

        # 3. Initialize and train the model
        model = LSTM(
            vocab_size,
            self.config.LSTM_EMBEDDING_DIM,
            self.config.LSTM_HIDDEN_DIM,
            self.config.LSTM_NUM_LAYERS
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LSTM_LEARNING_RATE)

        print(f"  Training LSTM model for {self.config.LSTM_EPOCHS} epochs...")
        model.train()
        for epoch in range(self.config.LSTM_EPOCHS):
            epoch_loss = 0.0
            num_batches = 0
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{self.config.LSTM_EPOCHS}", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"  Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        # 4. Generate embeddings for each protein
        print("  Generating per-protein embeddings using the trained LSTM...")
        model.eval()
        protein_embeddings = {}
        for pid, seq_text in tqdm(sequences, desc="  Generating Embeddings"):
            if not seq_text: continue
            input_tensor = torch.tensor([[char_to_idx[c] for c in seq_text if c in char_to_idx]]).to(self.device)
            if input_tensor.nelement() == 0: continue
            embedding = model.get_embedding(input_tensor)
            protein_embeddings[pid] = embedding.squeeze(0).cpu().numpy()

        # 5. Save embeddings
        # CRITICAL FIX: Use the correct output directory for LSTM embeddings.
        output_path = self.config.RESULTS_LSTM_EMBEDDINGS_DIR / "lstm_generated_embeddings.h5"
        DataUtils.write_h5(protein_embeddings, output_path, "Writing LSTM Embeddings")
        print(f"\nSUCCESS: LSTM embeddings saved to: {output_path}")
        return str(output_path)
