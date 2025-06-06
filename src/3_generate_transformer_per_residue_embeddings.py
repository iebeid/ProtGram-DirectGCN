import os
import time
import gc
import glob
from typing import List, Tuple, Dict, Iterator

import numpy as np
import h5py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, T5Tokenizer
from tqdm.auto import tqdm
from sklearn.decomposition import PCA

# --- User Configuration ---
DATASET_TAG = "uniref50_ProtBERT_per_protein"

# Input FASTA files
FASTA_INPUT_DIR = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/fasta/uniref50/"

# Output Directory for the final HDF5 files
OUTPUT_EMBEDDINGS_DIR = "C:/tmp/Models/protein_embeddings_from_transformer/"

# Transformer Models to Use
MODEL_CONFIGS = [{"name": "ProtBERT", "hf_id": "Rostlab/prot_bert", "is_t5": False, "batch_size_multiplier": 1},
    # {"name": "ProtT5_XL", "hf_id": "Rostlab/prot_t5_xl_uniref50", "is_t5": True, "batch_size_multiplier": 0.25}
]

# --- Processing Parameters ---
# Processing options
DEFAULT_MAX_LENGTH = 1024
BASE_BATCH_SIZE = 16

# Pooling and PCA options
POOLING_STRATEGY = 'mean'  # 'mean', 'sum', or 'max'
APPLY_PCA = True
PCA_TARGET_DIM = 64
SEED = 42

# Performance Options
ENABLE_MIXED_PRECISION = True
ENABLE_XLA = False


# --- End User Configuration ---


def fast_fasta_parser(fasta_filepath: str) -> Iterator[Tuple[str, str]]:
    """Efficiently parses a FASTA file, yielding one sequence at a time."""
    fasta_filepath = os.path.normpath(fasta_filepath)
    protein_id = None
    sequence_parts = []
    try:
        with open(fasta_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if protein_id and sequence_parts:
                        yield protein_id, "".join(sequence_parts)
                    header = line[1:]
                    parts = header.split('|')
                    protein_id = parts[1] if len(parts) > 1 and parts[1] else header.split()[0]
                    sequence_parts = []
                else:
                    sequence_parts.append(line.replace(" ", "").upper())
            if protein_id and sequence_parts:
                yield protein_id, "".join(sequence_parts)
    except FileNotFoundError:
        print(f"Warning: FASTA file not found: {fasta_filepath}")


def get_model_inference_function(model: tf.keras.Model, is_t5: bool, use_xla: bool) -> tf.types.experimental.GenericFunction:
    """Creates a model inference function, optionally compiled with XLA."""

    def model_call(inputs_dict_tf):
        if is_t5:
            num_seqs = tf.shape(inputs_dict_tf['input_ids'])[0]
            decoder_start_id = model.config.decoder_start_token_id or 0
            decoder_input_ids = tf.fill((num_seqs, 1), tf.cast(decoder_start_id, inputs_dict_tf['input_ids'].dtype))
            return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'], decoder_input_ids=decoder_input_ids)
        else:
            return model(inputs_dict_tf)

    if use_xla:
        return tf.function(model_call, jit_compile=True)
    return tf.function(model_call)


def extract_and_pool_embedding(raw_embedding: np.ndarray, original_seq_len: int, is_t5: bool, strategy: str) -> np.ndarray:
    """Extracts per-residue vectors and immediately pools them into a per-protein vector."""
    # Step 1: Slice to get per-residue embeddings (handling special tokens)
    if is_t5:
        residue_vectors = raw_embedding[:original_seq_len, :]
    else:
        residue_vectors = raw_embedding[1:original_seq_len + 1, :]

    # Step 2: Pool the per-residue vectors
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


def apply_pca(embeddings_dict: Dict[str, np.ndarray], target_dim: int, seed: int) -> Dict[str, np.ndarray]:
    """Applies PCA to a dictionary of embeddings."""
    print(f"\nApplying PCA to reduce dimensions to {target_dim}...")
    protein_ids = list(embeddings_dict.keys())
    embedding_matrix = np.array([embeddings_dict[pid] for pid in protein_ids])

    n_samples, n_features = embedding_matrix.shape
    if n_samples <= target_dim or n_features <= target_dim:
        print(f"  Warning: PCA skipped. Not enough samples or features for the target dimension.")
        return embeddings_dict

    pca = PCA(n_components=target_dim, random_state=seed)
    reduced_embeddings = pca.fit_transform(embedding_matrix)
    print(f"  PCA complete. Explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    return {protein_ids[i]: reduced_embeddings[i] for i in range(len(protein_ids))}


def generate_protein_embeddings_for_model(config: Dict, fasta_files: List[str]):
    """Main workflow to generate and save embeddings for a single transformer model."""
    model_name = config["name"]
    hf_id = config["hf_id"]
    is_t5 = config["is_t5"]
    batch_size = max(1, int(BASE_BATCH_SIZE * config["batch_size_multiplier"]))

    print(f"\n--- Starting: {model_name} ({hf_id}) ---")
    print(f"Batch Size: {batch_size}, Max Length: {DEFAULT_MAX_LENGTH}, Pooling: {POOLING_STRATEGY}")

    all_protein_embeddings = {}
    model, tokenizer, inference_func = None, None, None

    try:
        # --- Load Model and Tokenizer ---
        print("Loading tokenizer and model...")
        tokenizer = T5Tokenizer.from_pretrained(hf_id) if is_t5 else AutoTokenizer.from_pretrained(hf_id)
        model = TFAutoModel.from_pretrained(hf_id, from_pt=True)
        inference_func = get_model_inference_function(model, is_t5, ENABLE_XLA)
        print("Model and tokenizer loaded.")

        # --- Process FASTA files in batches ---
        for fasta_path in tqdm(fasta_files, desc=f"Files for {model_name}"):
            batch_sequences, batch_ids = [], []
            for prot_id, sequence in fast_fasta_parser(fasta_path):
                if not sequence: continue
                batch_sequences.append(sequence)
                batch_ids.append(prot_id)

                if len(batch_sequences) >= batch_size:
                    inputs_tf = tokenizer(batch_sequences, padding="max_length", truncation=True, return_tensors="tf", max_length=DEFAULT_MAX_LENGTH)
                    model_outputs = inference_func(inputs_tf)
                    raw_batch = (model_outputs.encoder_last_hidden_state if is_t5 else model_outputs.last_hidden_state).numpy()

                    for i in range(len(batch_ids)):
                        pooled_vec = extract_and_pool_embedding(raw_batch[i], len(batch_sequences[i]), is_t5, POOLING_STRATEGY)
                        if pooled_vec.size > 0:
                            all_protein_embeddings[batch_ids[i]] = pooled_vec

                    batch_sequences, batch_ids = [], []

            # Process final batch
            if batch_sequences:
                inputs_tf = tokenizer(batch_sequences, padding="max_length", truncation=True, return_tensors="tf", max_length=DEFAULT_MAX_LENGTH)
                model_outputs = inference_func(inputs_tf)
                raw_batch = (model_outputs.encoder_last_hidden_state if is_t5 else model_outputs.last_hidden_state).numpy()
                for i in range(len(batch_ids)):
                    pooled_vec = extract_and_pool_embedding(raw_batch[i], len(batch_sequences[i]), is_t5, POOLING_STRATEGY)
                    if pooled_vec.size > 0:
                        all_protein_embeddings[batch_ids[i]] = pooled_vec

        print(f"\nGenerated {len(all_protein_embeddings)} total protein embeddings.")

        # --- Apply PCA (optional) ---
        final_embeddings = all_protein_embeddings
        if APPLY_PCA and len(all_protein_embeddings) > 1:
            final_embeddings = apply_pca(all_protein_embeddings, PCA_TARGET_DIM, SEED)

        # --- Save to HDF5 ---
        output_filename = f"{model_name.replace(' ', '_')}_per-protein_{POOLING_STRATEGY}_{DATASET_TAG}"
        output_filename += f"_pca{PCA_TARGET_DIM}.h5" if APPLY_PCA else ".h5"
        output_path = os.path.join(OUTPUT_EMBEDDINGS_DIR, output_filename)

        print(f"Saving final embeddings to: {output_path}")
        with h5py.File(output_path, 'w') as hf:
            for prot_id, embedding in final_embeddings.items():
                hf.create_dataset(prot_id.replace('/', '_'), data=embedding)
            hf.attrs.update({'model_name': model_name, 'huggingface_id': hf_id, 'pooling_strategy': POOLING_STRATEGY, 'pca_applied': APPLY_PCA, 'final_embedding_dim': next(iter(final_embeddings.values())).shape[0]})
        print("Save complete.")

    except Exception as e:
        print(f"\nFATAL ERROR during processing for model {model_name}: {e}")
    finally:
        # --- Clean up memory ---
        print(f"Clearing {model_name} from memory...")
        del model, tokenizer, inference_func, all_protein_embeddings
        gc.collect()
        tf.keras.backend.clear_session()
        print("Memory cleared.")


def main():
    """Main script execution function."""
    print("--- Direct Per-Protein Embedding Generation using Transformers ---")
    start_time = time.time()
    os.makedirs(OUTPUT_EMBEDDINGS_DIR, exist_ok=True)

    if ENABLE_MIXED_PRECISION:
        print("Enabling TensorFlow Mixed Precision (mixed_float16).")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    fasta_files = [os.path.normpath(f) for f in glob.glob(os.path.join(FASTA_INPUT_DIR, '*.fasta'))]
    if not fasta_files:
        print(f"Error: No FASTA files found in {FASTA_INPUT_DIR}. Exiting.")
        return
    print(f"Found {len(fasta_files)} FASTA file(s) to process.")

    for config in MODEL_CONFIGS:
        generate_protein_embeddings_for_model(config, fasta_files)

    print(f"\n--- All models processed. Script finished in {time.time() - start_time:.2f} seconds. ---")


if __name__ == "__main__":
    main()
