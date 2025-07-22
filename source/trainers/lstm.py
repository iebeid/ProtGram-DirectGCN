# ==============================================================================
# MODULE: trainers/lstm.py
# PURPOSE: Trainer for a character-level LSTM model to generate protein embeddings.
# VERSION: 2.0 (Corrected output path and used configured sequence length)
# AUTHOR: Islam Ebeid
# ==============================================================================

import numpy as np
import torch
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence, to_categorical
from tqdm.auto import tqdm

from configuration.config import Config
from source.models.ml.lstm import LSTM
from source.utils.data import DataUtils, FastaUtils


# ==============================================================================
# NEW: Data Generator to prevent Out-of-Memory errors
# ==============================================================================
class LstmCorpusGenerator(Sequence):
    """
    Generates batches of data for the LSTM model on-the-fly from a FASTA file.
    """

    def __init__(self, fasta_paths, batch_size, seq_len, vocab_size, char_to_int):
        self.fasta_paths = fasta_paths
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.char_to_int = char_to_int

        # Create one long sequence of all proteins
        print("  [Generator] Concatenating sequences from FASTA files...")
        self.text = "".join([seq for _, seq in FastaUtils.parse_sequences(self.fasta_paths)])
        self.text_len = len(self.text)
        print(f"  [Generator] Corpus created with {self.text_len:,} characters.")

        # Calculate the number of sequences we can create
        self.num_sequences = (self.text_len - seq_len) // 1

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return (self.num_sequences) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data."""
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        batch_x = []
        batch_y = []

        for i in range(start_idx, end_idx):
            if i + self.seq_len >= self.text_len:
                continue
            # Input sequence
            in_seq = self.text[i: i + self.seq_len]
            # Output character
            out_char = self.text[i + self.seq_len]

            batch_x.append([self.char_to_int[char] for char in in_seq])
            batch_y.append(self.char_to_int[out_char])

        # Convert to numpy arrays and one-hot encode the output
        X = np.array(batch_x)
        y = to_categorical(batch_y, num_classes=self.vocab_size)

        return X, y


class LSTMBasedEmbedder:
    """
    Trains a character-level LSTM on a next-character prediction task and then
    uses the trained model to generate a single embedding vector for each
    protein sequence.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = config
        self.output_dir = self.config.RESULTS_LSTM_EMBEDDINGS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.char_to_int = {}
        self.int_to_char = {}
        self.vocab_size = 0

    def _prepare_corpus(self):
        """Prepares the character mapping from the FASTA data."""
        print("  Preparing LSTM corpus and character mappings...")
        self.sequences = FastaUtils.parse_sequences(self.config.SEQUENCE_FILE_PATHS)
        self.chars = sorted(list(set("".join(seq for _, seq in self.sequences))))

        self.char_to_int = {c: i for i, c in enumerate(self.chars)}
        self.int_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        print(f"  Vocabulary size: {self.vocab_size}")

    def _build_model(self):
        """Builds the LSTM model for next-character prediction."""
        model = Sequential([
            Embedding(self.vocab_size, self.config.LSTM_EMBEDDING_DIM, input_length=self.config.LSTM_TRAIN_SEQ_LEN),
            LSTM(self.config.LSTM_HIDDEN_DIM, return_sequences=True, num_layers=self.config.LSTM_NUM_LAYERS),
            LSTM(self.config.LSTM_HIDDEN_DIM),
            Dense(self.vocab_size, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        self.model = model

    def run(self):
        DataUtils.print_header("PIPELINE: Training LSTM & Generating Embeddings")

        self._prepare_corpus()
        self._build_model()

        # --- MODIFICATION START: Use the data generator ---
        print(f"  Training LSTM model for {self.config.LSTM_EPOCHS} epochs using a data generator...")

        # Instantiate the generator
        training_generator = LstmCorpusGenerator(
            fasta_paths=self.config.SEQUENCE_FILE_PATHS,
            batch_size=self.config.LSTM_BATCH_SIZE,
            seq_len=self.config.LSTM_TRAIN_SEQ_LEN,
            vocab_size=self.vocab_size,
            char_to_int=self.char_to_int
        )

        # Fit the model using the generator
        self.model.fit(
            training_generator,
            epochs=self.config.LSTM_EPOCHS,
            verbose=1
        )
        # --- MODIFICATION END ---

        print("  LSTM training complete. Generating embeddings...")

        # 4. Generate embeddings for each protein
        print("  Generating per-protein embeddings using the trained LSTM...")
        self.model.eval()
        protein_embeddings = {}
        for pid, seq_text in tqdm(self.sequences, desc="  Generating Embeddings"):
            if not seq_text: continue
            input_tensor = torch.tensor([[self.char_to_int[c] for c in seq_text if c in self.char_to_int]]).to(self.device)
            if input_tensor.nelement() == 0: continue
            embedding = self.model.get_embedding(input_tensor)
            protein_embeddings[pid] = embedding.squeeze(0).cpu().numpy()

        # 5. Save embeddings
        # CRITICAL FIX: Use the correct output directory for LSTM embeddings.
        output_path = self.config.RESULTS_LSTM_EMBEDDINGS_DIR / "lstm_generated_embeddings.h5"
        DataUtils.write_h5(protein_embeddings, output_path, "Writing LSTM Embeddings")
        print(f"\nSUCCESS: LSTM embeddings saved to: {output_path}")
        return str(output_path)
