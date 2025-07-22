# source/trainers/lstm.py

# ==============================================================================
# MODULE: trainers/lstm.py
# PURPOSE: Trainer for a character-level LSTM model to generate protein embeddings.
# VERSION: 5.0 (Corrected downsampling logic to respect LSTM-specific config)
# AUTHOR: Islam Ebeid
# ==============================================================================

import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import Sequence, to_categorical
from tqdm.auto import tqdm

from configuration.config import Config
from source.utils.data import DataUtils, FastaUtils


class LstmCorpusGenerator(Sequence):
    """
    Generates batches of data for the LSTM model on-the-fly.
    This version now accepts a list of sequences directly to avoid re-reading files.
    """

    # --- FIX: Accept a list of sequences instead of file paths ---
    def __init__(self, sequences: List[Tuple[str, str]], batch_size: int, seq_len: int, step: int, vocab_size: int,
                 char_to_int: dict):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.step = step
        self.vocab_size = vocab_size
        self.char_to_int = char_to_int

        print("  [Generator] Concatenating sequences for corpus...")
        # --- FIX: Build corpus from the provided list of sequences ---
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
        """
        Prepares the corpus by loading sequences and applying LSTM-specific downsampling.
        """
        print("  Preparing LSTM corpus and character mappings...")
        # Load all sequences from the file path provided by the main pipeline
        all_loaded_sequences = list(FastaUtils.parse_sequences(self.config.SEQUENCE_FILE_PATHS))

        # --- FIX: Apply LSTM-specific downsampling if configured ---
        should_downsample = self.config.LSTM_DOWNSAMPLE_FRACTION and 0 < self.config.LSTM_DOWNSAMPLE_FRACTION < 1.0
        if should_downsample:
            sample_size = int(len(all_loaded_sequences) * self.config.LSTM_DOWNSAMPLE_FRACTION)
            print(f"  Applying LSTM-specific downsampling: {self.config.LSTM_DOWNSAMPLE_FRACTION:.1%}")
            print(f"    - Sampling {sample_size} of {len(all_loaded_sequences)} sequences.")
            self.sequences = random.sample(all_loaded_sequences, sample_size)
        else:
            self.sequences = all_loaded_sequences
        # --- END FIX ---

        all_chars = sorted(list(set("".join(seq for _, seq in self.sequences))))
        self.char_to_int = {c: i for i, c in enumerate(all_chars)}
        self.int_to_char = {i: c for i, c in enumerate(all_chars)}
        self.vocab_size = len(all_chars)
        print(f"  Vocabulary size: {self.vocab_size}")

    def _build_model(self):
        """Builds the Keras LSTM model for next-character prediction."""
        model = Sequential([
            Embedding(self.config.LSTM_EMBEDDING_DIM, self.config.LSTM_EMBEDDING_DIM,
                      input_length=None, name="embedding_layer"),
            LSTM(self.config.LSTM_HIDDEN_DIM, return_sequences=True, name="lstm_layer_1", recurrent_dropout=0.1),
            LSTM(self.config.LSTM_HIDDEN_DIM, return_sequences=True, name="lstm_embedding_layer", recurrent_dropout=0.1),
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
        # --- FIX: Pass the downsampled list of sequences directly to the generator ---
        training_generator = LstmCorpusGenerator(
            sequences=self.sequences,  # Pass the in-memory list
            batch_size=self.config.LSTM_BATCH_SIZE,
            seq_len=self.config.LSTM_TRAIN_SEQ_LEN,
            step=self.config.LSTM_TRAIN_STEP,
            vocab_size=self.vocab_size,
            char_to_int=self.char_to_int
        )

        # We need to modify the training model slightly for the generator output shape
        train_model_input = tf.keras.Input(shape=(self.config.LSTM_TRAIN_SEQ_LEN,))
        x = self.model.get_layer('embedding_layer')(train_model_input)
        x = self.model.get_layer('lstm_layer_1')(x)
        x = self.model.get_layer('lstm_embedding_layer')(x)
        x = x[:, -1, :]
        train_model_output = self.model.get_layer('output_dense_layer')(x)
        train_model = Model(inputs=train_model_input, outputs=train_model_output)
        train_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        train_model.fit(training_generator, epochs=self.config.LSTM_EPOCHS, verbose=1)

        print("  LSTM training complete. Generating embeddings...")

        print("  Building inference model to extract full-sequence hidden states...")
        inf_input = tf.keras.Input(shape=(None,), dtype=tf.int32)
        x = self.model.get_layer('embedding_layer')(inf_input)
        x = self.model.get_layer('lstm_layer_1')(x)
        embedding_layer_output = self.model.get_layer('lstm_embedding_layer')(x)
        inference_model = Model(inputs=inf_input, outputs=embedding_layer_output)
        inference_model.summary()

        print("  Generating per-protein embeddings using mean pooling...")
        protein_embeddings = {}
        # Use self.sequences here as it's the correctly downsampled list
        for pid, seq_text in tqdm(self.sequences, desc="  Generating Embeddings"):
            if not seq_text: continue
            tokenized_seq = [self.char_to_int[c] for c in seq_text if c in self.char_to_int]
            if not tokenized_seq: continue

            input_tensor = tf.constant([tokenized_seq], dtype=tf.int32)
            all_hidden_states = inference_model.predict(input_tensor, verbose=0)
            pooled_embedding = tf.reduce_mean(all_hidden_states, axis=1).numpy().squeeze(0)
            protein_embeddings[pid] = pooled_embedding

        output_path = self.config.RESULTS_LSTM_EMBEDDINGS_DIR / "lstm_generated_embeddings.h5"
        DataUtils.write_h5(protein_embeddings, output_path, "Writing LSTM Embeddings")
        print(f"\nSUCCESS: LSTM embeddings saved to: {output_path}")
        return str(output_path)
