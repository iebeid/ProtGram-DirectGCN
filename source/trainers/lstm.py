# ==============================================================================
# MODULE: trainers/lstm.py
# PURPOSE: Trainer for a character-level LSTM model to generate protein embeddings.
# VERSION: 2.1 (Corrected Keras/PyTorch library mix-up and inference logic)
# AUTHOR: Islam Ebeid
# ==============================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import Sequence, to_categorical
from tqdm.auto import tqdm

from configuration.config import Config
from source.utils.data import DataUtils, FastaUtils


# ==============================================================================
# Data Generator (No changes needed here, it is correct)
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

        print("  [Generator] Concatenating sequences from FASTA files...")
        self.text = "".join([seq for _, seq in FastaUtils.parse_sequences(self.fasta_paths)])
        self.text_len = len(self.text)
        print(f"  [Generator] Corpus created with {self.text_len:,} characters.")
        self.num_sequences = (self.text_len - seq_len)

    def __len__(self):
        return self.num_sequences // self.batch_size

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_x, batch_y = [], []
        for i in range(start_idx, end_idx):
            if i + self.seq_len >= self.text_len: continue
            in_seq = self.text[i: i + self.seq_len]
            out_char = self.text[i + self.seq_len]
            batch_x.append([self.char_to_int[char] for char in in_seq])
            batch_y.append(self.char_to_int[out_char])
        return np.array(batch_x), to_categorical(batch_y, num_classes=self.vocab_size)


class LSTMBasedEmbedder:
    """
    Trains a character-level LSTM and uses it to generate protein embeddings.
    """

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = self.config.RESULTS_LSTM_EMBEDDINGS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.sequences = []
        self.char_to_int = {}
        self.int_to_char = {}
        self.vocab_size = 0

    def _prepare_corpus(self):
        """Prepares the character mapping from the FASTA data."""
        print("  Preparing LSTM corpus and character mappings...")
        # FIX: Convert iterator to list to allow multiple iterations over the sequences.
        self.sequences = list(FastaUtils.parse_sequences(self.config.SEQUENCE_FILE_PATHS))
        all_chars = sorted(list(set("".join(seq for _, seq in self.sequences))))

        self.char_to_int = {c: i for i, c in enumerate(all_chars)}
        self.int_to_char = {i: c for i, c in enumerate(all_chars)}
        self.vocab_size = len(all_chars)
        print(f"  Vocabulary size: {self.vocab_size}")

    def _build_model(self):
        """Builds the Keras LSTM model for next-character prediction."""
        # FIX: Add names to layers for easy reference during inference model creation.
        model = Sequential([
            Embedding(self.vocab_size, self.config.LSTM_EMBEDDING_DIM,
                      input_length=self.config.LSTM_TRAIN_SEQ_LEN, name="embedding_layer"),
            LSTM(self.config.LSTM_HIDDEN_DIM, return_sequences=True, name="lstm_layer_1"),
            LSTM(self.config.LSTM_HIDDEN_DIM, name="lstm_embedding_layer"),
            Dense(self.vocab_size, activation='softmax', name="output_dense_layer")
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        self.model = model

    def run(self):
        DataUtils.print_header("PIPELINE: Training LSTM & Generating Embeddings")
        self._prepare_corpus()
        self._build_model()

        print(f"  Training LSTM model for {self.config.LSTM_EPOCHS} epochs using a data generator...")
        training_generator = LstmCorpusGenerator(
            fasta_paths=self.config.SEQUENCE_FILE_PATHS,
            batch_size=self.config.LSTM_BATCH_SIZE,
            seq_len=self.config.LSTM_TRAIN_SEQ_LEN,
            vocab_size=self.vocab_size,
            char_to_int=self.char_to_int
        )
        self.model.fit(training_generator, epochs=self.config.LSTM_EPOCHS, verbose=1)

        print("  LSTM training complete. Generating embeddings...")

        # --- MINIMAL FIX: Correct Keras inference logic ---
        # 1. Create a new model for inference that outputs the hidden state of the final LSTM layer.
        #    This model re-uses the layers and weights from the trained model.
        print("  Building inference model to extract LSTM hidden states...")
        embedding_layer_output = self.model.get_layer('lstm_embedding_layer').output
        inference_model = Model(inputs=self.model.input, outputs=embedding_layer_output)
        inference_model.summary()

        # 2. Generate embeddings for each protein using the new Keras inference model.
        print("  Generating per-protein embeddings using the inference model...")
        protein_embeddings = {}
        for pid, seq_text in tqdm(self.sequences, desc="  Generating Embeddings"):
            if not seq_text: continue
            tokenized_seq = [self.char_to_int[c] for c in seq_text if c in self.char_to_int]
            if not tokenized_seq: continue

            # Pad the sequence to the required training length for the model input
            padded_seq = tf.keras.preprocessing.sequence.pad_sequences(
                [tokenized_seq], maxlen=self.config.LSTM_TRAIN_SEQ_LEN, padding='pre'
            )

            # Get the embedding from the inference model
            embedding = inference_model.predict(padded_seq, verbose=0)
            protein_embeddings[pid] = embedding.squeeze(0)  # Remove batch dimension
        # --- END FIX ---

        # 3. Save embeddings
        output_path = self.config.RESULTS_LSTM_EMBEDDINGS_DIR / "lstm_generated_embeddings.h5"
        DataUtils.write_h5(protein_embeddings, output_path, "Writing LSTM Embeddings")
        print(f"\nSUCCESS: LSTM embeddings saved to: {output_path}")
        return str(output_path)
