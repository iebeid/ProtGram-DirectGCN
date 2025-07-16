# ==============================================================================
# MODULE: trainers/lstm.py
# PURPOSE: Trainer for a character-level LSTM model to generate protein embeddings.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import numpy as np

from configuration.config import Config
from source.utils.data import DataLoader as SeqLoader
from source.utils.data import DataUtils
from source.utils.models import EmbeddingProcessor


class AminoAcidDataset(Dataset):
    """Dataset for preparing sequences for a next-character prediction task."""
    def __init__(self, sequences, seq_len, char_to_idx):
        self.seq_len = seq_len
        self.char_to_idx = char_to_idx
        self.idx_to_char = {i: c for c, i in char_to_idx.items()}
        
        self.X = []
        self.y = []

        for seq in sequences:
            text = seq[1] # seq is a tuple (id, text)
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

class LSTMAminoAcidPredictor(nn.Module):
    """An LSTM model to predict the next amino acid."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        # We only need the output of the last time step for prediction
        last_time_step_out = lstm_out[:, -1, :]
        out = self.fc(last_time_step_out)
        return out

    def get_embedding(self, x):
        """Gets the protein embedding (output of LSTM's last hidden state)."""
        with torch.no_grad():
            x = self.embedding(x)
            _, (hidden, _) = self.lstm(x)
            # Return the hidden state of the last layer
            return hidden[-1]


class LSTMBasedEmbedder:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        """Main function to train LSTM and generate embeddings."""
        DataUtils.print_header("PIPELINE: Training LSTM & Generating Embeddings")
        
        # 1. Load data and create vocabulary
        sequences = list(SeqLoader.parse_sequences(self.config.SEQUENCE_FILE_PATHS))
        chars = sorted(list(set("".join(s[1] for s in sequences))))
        char_to_idx = {c: i for i, c in enumerate(chars)}
        vocab_size = len(chars)

        # 2. Create Dataset and DataLoader
        # Using a fixed sequence length for training
        train_seq_len = 50 
        dataset = AminoAcidDataset(sequences, train_seq_len, char_to_idx)
        if not dataset:
            print("  No training data generated for LSTM. Sequence lengths might be too short. Skipping.")
            return None
        dataloader = DataLoader(dataset, batch_size=self.config.LSTM_BATCH_SIZE, shuffle=True)

        # 3. Initialize and train the model
        model = LSTMAminoAcidPredictor(
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
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.LSTM_EPOCHS}", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f"  Epoch {epoch+1} Loss: {loss.item():.4f}")

        # 4. Generate embeddings for each protein
        print("  Generating per-protein embeddings using the trained LSTM...")
        model.eval()
        protein_embeddings = {}
        for pid, seq_text in tqdm(sequences, desc="  Generating Embeddings"):
            if not seq_text: continue
            input_tensor = torch.tensor([[char_to_idx[c] for c in seq_text]]).to(self.device)
            embedding = model.get_embedding(input_tensor)
            protein_embeddings[pid] = embedding.squeeze(0).cpu().numpy()

        # 5. Save embeddings
        output_path = self.config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR / "lstm_generated_embeddings.h5"
        self._write_h5(protein_embeddings, output_path, "Writing LSTM Embeddings")
        print(f"\nSUCCESS: LSTM embeddings saved to: {output_path}")
        return str(output_path)

    def _write_h5(self, embeddings_dict, path, desc):
        """Helper to write embeddings to HDF5."""
        path.parent.mkdir(parents=True, exist_ok=True)
        import h5py
        with h5py.File(path, 'w') as hf:
            for key, value in tqdm(embeddings_dict.items(), desc=f"  {desc}"):
                if value is not None:
                    hf.create_dataset(key, data=value)