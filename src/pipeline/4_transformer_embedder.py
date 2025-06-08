# ==============================================================================
# MODULE: pipeline/4_transformer_embedder.py
# PURPOSE: Generates per-protein embeddings using pre-trained Transformer
#          models from Hugging Face.
# ==============================================================================

import os
import gc
import glob
import time
from typing import List, Tuple, Dict, Iterator

import numpy as np
import h5py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, T5Tokenizer
from tqdm.auto import tqdm

# Import from our new project structure
from config import Config
from utils.data_loader import fast_fasta_parser
from utils.embedding_tools import apply_pca


def _get_model_inference_function(model: tf.keras.Model, is_t5: bool, use_xla: bool) -> tf.types.experimental.GenericFunction:
    """Creates a model inference function, optionally compiled with XLA for performance."""

    def model_call(inputs_dict_tf):
        if is_t5:
            # T5 models require a decoder_input_ids to be passed
            num_seqs = tf.shape(inputs_dict_tf['input_ids'])[0]
            decoder_start_id = model.config.decoder_start_token_id or 0
            decoder_input_ids = tf.fill((num_seqs, 1), tf.cast(decoder_start_id, inputs_dict_tf['input_ids'].dtype))
            return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'], decoder_input_ids=decoder_input_ids)
        else:
            return model(inputs_dict_tf)

    if use_xla:
        return tf.function(model_call, jit_compile=True)
    return tf.function(model_call)


def _extract_and_pool_embedding(raw_embedding: np.ndarray, original_seq_len: int, is_t5: bool, strategy: str) -> np.ndarray:
    """Extracts per-residue vectors and pools them into a per-protein vector."""
    # Slice to get per-residue embeddings, handling special tokens [CLS] and [SEP]
    if is_t5:
        residue_vectors = raw_embedding[:original_seq_len, :]
    else:  # For BERT-like models
        residue_vectors = raw_embedding[1:original_seq_len + 1, :]

    if residue_vectors.shape[0] == 0:
        return np.array([])

    if strategy == 'mean':
        return np.mean(residue_vectors, axis=0)
    elif strategy == 'sum':
        return np.sum(residue_vectors, axis=0)
    elif strategy == 'max':
        return np.max(residue_vectors, axis=0)
    else:
        raise ValueError(f"Unknown pooling strategy: {strategy}")


def _generate_embeddings_for_single_model(model_config: Dict, fasta_files: List[str], config: Config):
    """Main workflow to generate and save embeddings for a single transformer model."""
    model_name = model_config["name"]
    hf_id = model_config["hf_id"]
    is_t5 = model_config["is_t5"]
    batch_size = max(1, int(config.TRANSFORMER_BASE_BATCH_SIZE * model_config["batch_size_multiplier"]))

    print(f"\n--- Starting: {model_name} ({hf_id}) ---")
    print(f"  Batch Size: {batch_size}, Max Length: {config.TRANSFORMER_MAX_LENGTH}, Pooling: {config.TRANSFORMER_POOLING_STRATEGY}")

    all_protein_embeddings = {}
    model, tokenizer, inference_func = None, None, None

    try:
        # Load Model and Tokenizer
        print("  Loading tokenizer and model...")
        tokenizer_class = T5Tokenizer if is_t5 else AutoTokenizer
        tokenizer = tokenizer_class.from_pretrained(hf_id)
        model = TFAutoModel.from_pretrained(hf_id, from_pt=True)
        inference_func = _get_model_inference_function(model, is_t5, False)  # XLA can be unstable, keeping it off
        print("  Model and tokenizer loaded.")

        # Process FASTA files in batches
        for fasta_path in tqdm(fasta_files, desc=f"FASTA Files for {model_name}"):
            batch_sequences, batch_ids = [], []
            for prot_id, sequence in fast_fasta_parser(fasta_path):
                if not sequence: continue
                # Replace rare amino acids
                sequence = sequence.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
                batch_sequences.append(" ".join(list(sequence)))  # Add spaces for tokenizer
                batch_ids.append(prot_id)

                if len(batch_sequences) >= batch_size:
                    inputs = tokenizer(batch_sequences, padding="longest", truncation=True, return_tensors="tf", max_length=config.TRANSFORMER_MAX_LENGTH)
                    outputs = inference_func(inputs)
                    raw_batch = (outputs.encoder_last_hidden_state if is_t5 else outputs.last_hidden_state).numpy()

                    for i in range(len(batch_ids)):
                        seq_len = len(batch_sequences[i].replace(" ", ""))
                        pooled_vec = _extract_and_pool_embedding(raw_batch[i], seq_len, is_t5, config.TRANSFORMER_POOLING_STRATEGY)
                        if pooled_vec.size > 0: all_protein_embeddings[batch_ids[i]] = pooled_vec
                    batch_sequences, batch_ids = [], []

            # Process final leftover batch
            if batch_sequences:
                inputs = tokenizer(batch_sequences, padding="longest", truncation=True, return_tensors="tf", max_length=config.TRANSFORMER_MAX_LENGTH)
                outputs = inference_func(inputs)
                raw_batch = (outputs.encoder_last_hidden_state if is_t5 else outputs.last_hidden_state).numpy()
                for i in range(len(batch_ids)):
                    seq_len = len(batch_sequences[i].replace(" ", ""))
                    pooled_vec = _extract_and_pool_embedding(raw_batch[i], seq_len, is_t5, config.TRANSFORMER_POOLING_STRATEGY)
                    if pooled_vec.size > 0: all_protein_embeddings[batch_ids[i]] = pooled_vec

        print(f"\n  Generated {len(all_protein_embeddings)} total protein embeddings for {model_name}.")

        # Apply PCA (optional)
        final_embeddings = all_protein_embeddings
        if config.APPLY_PCA_TO_TRANSFORMER and len(all_protein_embeddings) > 1:
            final_embeddings = apply_pca(all_protein_embeddings, config.PCA_TARGET_DIMENSION, config.RANDOM_STATE)

        # Save to HDF5
        output_filename = f"{model_name}_{config.TRANSFORMER_POOLING_STRATEGY}"
        output_filename += f"_pca{config.PCA_TARGET_DIMENSION}.h5" if config.APPLY_PCA_TO_TRANSFORMER else "_full_dim.h5"
        output_path = os.path.join(config.TRANSFORMER_EMBEDDINGS_DIR, output_filename)

        print(f"  Saving final embeddings to: {output_path}")
        with h5py.File(output_path, 'w') as hf:
            for prot_id, embedding in final_embeddings.items():
                hf.create_dataset(prot_id.replace('/', '_'), data=embedding)
        print("  Save complete.")

    except Exception as e:
        print(f"\nFATAL ERROR during processing for model {model_name}: {e}")
    finally:
        print(f"  Clearing {model_name} from memory...")
        del model, tokenizer, inference_func, all_protein_embeddings;
        gc.collect();
        tf.keras.backend.clear_session()


# --- Main Orchestration Function for this Module ---
def run_transformer_embedding_generation(config: Config):
    """
    The main entry point for the Transformer embedding generation pipeline step.
    """
    print("\n" + "=" * 80);
    print("### PIPELINE STEP: Generating Embeddings from Transformers ###");
    print("=" * 80)
    os.makedirs(config.TRANSFORMER_EMBEDDINGS_DIR, exist_ok=True)

    if tf.config.list_physical_devices('GPU'):
        print("Enabling TensorFlow Mixed Precision (mixed_float16).")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    fasta_files = [os.path.normpath(f) for f in glob.glob(os.path.join(config.TRANSFORMER_INPUT_FASTA_DIR, '*.fasta')) + glob.glob(os.path.join(config.TRANSFORMER_INPUT_FASTA_DIR, '*.fa'))]
    if not fasta_files:
        print(f"Error: No FASTA files found in {config.TRANSFORMER_INPUT_FASTA_DIR}. Skipping Transformer step.");
        return
    print(f"Found {len(fasta_files)} FASTA file(s) to process.")

    for model_config in config.TRANSFORMER_MODELS_TO_RUN:
        _generate_embeddings_for_single_model(model_config, fasta_files, config)

    print("\n### Transformer PIPELINE STEP FINISHED ###")