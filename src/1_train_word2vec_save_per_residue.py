import os
import time
import gc
import glob
from collections.abc import Iterator
from typing import List, Tuple, Dict

import numpy as np
import h5py
from gensim.models import Word2Vec
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
import concurrent.futures

# --- User Configuration ---

# General settings
RUN_TAG = "UniRef50_Word2Vec_Pooled_v1"
BASE_OUTPUT_DIR = "C:/tmp/Models/protein_embeddings/"

# FASTA input configuration
FASTA_INPUT_DIR = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/fasta/uniref50/"

# Word2Vec Hyperparameters
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 1
W2V_EPOCHS = 1
W2V_WORKERS = max(1, os.cpu_count() - 2 if os.cpu_count() else 1)
W2V_SG = 1  # Use Skip-Gram

# Pooling and PCA settings
POOLING_STRATEGY = 'mean'  # 'mean', 'sum', or 'max'
APPLY_PCA = True
PCA_TARGET_DIM = 64
SEED = 42


# --- End User Configuration ---


def fast_fasta_parser(fasta_filepath: str) -> Iterator[Tuple[str, str]]:
    """
    An efficient FASTA parser that reads one sequence at a time.

    Args:
        fasta_filepath: The path to the FASTA file.

    Yields:
        A tuple containing the protein ID and its sequence.
    """
    fasta_filepath = os.path.normpath(fasta_filepath)
    protein_id = None
    sequence_parts = []
    try:
        with open(fasta_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if protein_id and sequence_parts:
                        yield protein_id, "".join(sequence_parts)

                    # Extract a clean ID
                    header = line[1:]
                    parts = header.split('|')
                    if len(parts) > 1 and parts[1]:
                        protein_id = parts[1]
                    else:
                        protein_id = header.split()[0]
                    sequence_parts = []
                else:
                    sequence_parts.append(line.upper())

            if protein_id and sequence_parts:
                yield protein_id, "".join(sequence_parts)
    except FileNotFoundError:
        print(f"Error: FASTA file not found at {fasta_filepath}")


class FastaCorpus:
    """
    A memory-efficient corpus for Word2Vec training that reads from FASTA files.
    """

    def __init__(self, fasta_files: List[str]):
        self.fasta_files = fasta_files

    def __iter__(self) -> Iterator[List[str]]:
        for f_path in self.fasta_files:
            for _, sequence in fast_fasta_parser(f_path):
                yield list(sequence)


def pool_protein_embedding(sequence: str, w2v_model: Word2Vec, pooling_strategy: str = 'mean') -> np.ndarray:
    """
    Generates a per-protein embedding by pooling per-residue vectors.

    Args:
        sequence: The protein sequence.
        w2v_model: The trained Word2Vec model.
        pooling_strategy: The pooling method ('mean', 'sum', or 'max').

    Returns:
        The pooled per-protein embedding.
    """
    embedding_dim = w2v_model.vector_size

    sum_of_vectors = np.zeros(embedding_dim, dtype=np.float32)
    max_vector = np.full(embedding_dim, -np.inf, dtype=np.float32)
    valid_residues_count = 0

    for residue in sequence:
        if residue in w2v_model.wv:
            vec = w2v_model.wv[residue]
            sum_of_vectors += vec
            max_vector = np.maximum(max_vector, vec)
            valid_residues_count += 1

    if valid_residues_count == 0:
        return np.zeros(embedding_dim, dtype=np.float32)

    if pooling_strategy == 'mean':
        return sum_of_vectors / valid_residues_count
    elif pooling_strategy == 'sum':
        return sum_of_vectors
    elif pooling_strategy == 'max':
        return max_vector
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")


def main():
    """
    Main function to train Word2Vec, generate, and save pooled protein embeddings.
    """
    print(f"--- Protein Embedding Generation (Tag: {RUN_TAG}) ---")
    start_time = time.time()

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # --- 1. Find FASTA files ---
    fasta_files = [os.path.normpath(f) for f in glob.glob(os.path.join(FASTA_INPUT_DIR, '*.fasta'))]
    if not fasta_files:
        print(f"Error: No FASTA files found in {FASTA_INPUT_DIR}")
        return

    print(f"Found {len(fasta_files)} FASTA file(s).")

    # --- 2. Train Word2Vec model ---
    print("\nTraining Word2Vec model...")
    corpus = FastaCorpus(fasta_files)
    w2v_model = Word2Vec(sentences=corpus, vector_size=W2V_VECTOR_SIZE, window=W2V_WINDOW, min_count=W2V_MIN_COUNT, workers=W2V_WORKERS, sg=W2V_SG, epochs=W2V_EPOCHS)
    print("Word2Vec model training complete.")

    # --- 3. Generate and pool embeddings ---
    print("\nGenerating and pooling per-protein embeddings...")
    pooled_embeddings = {}

    for fasta_file in tqdm(fasta_files, desc="Processing FASTA files"):
        for protein_id, sequence in fast_fasta_parser(fasta_file):
            pooled_embeddings[protein_id] = pool_protein_embedding(sequence, w2v_model, POOLING_STRATEGY)

    print(f"Generated {len(pooled_embeddings)} pooled protein embeddings.")

    # --- 4. Apply PCA (optional) ---
    final_embeddings = pooled_embeddings
    if APPLY_PCA and len(pooled_embeddings) > 1:
        print(f"\nApplying PCA to reduce dimensions to {PCA_TARGET_DIM}...")

        ids = list(pooled_embeddings.keys())
        embedding_matrix = np.array([pooled_embeddings[pid] for pid in ids])

        pca = PCA(n_components=PCA_TARGET_DIM, random_state=SEED)
        reduced_embeddings = pca.fit_transform(embedding_matrix)

        final_embeddings = {ids[i]: reduced_embeddings[i] for i in range(len(ids))}
        print("PCA application complete.")

    # --- 5. Save final embeddings ---
    output_filename = f"{RUN_TAG}_pooled_embeddings.h5"
    if APPLY_PCA:
        output_filename = f"{RUN_TAG}_pooled_pca_{PCA_TARGET_DIM}_embeddings.h5"

    output_path = os.path.join(BASE_OUTPUT_DIR, output_filename)

    print(f"\nSaving final embeddings to: {output_path}")
    with h5py.File(output_path, 'w') as hf:
        for protein_id, embedding in final_embeddings.items():
            safe_id = protein_id.replace('/', '_')  # HDF5 doesn't like slashes in names
            hf.create_dataset(safe_id, data=embedding)

        hf.attrs['run_tag'] = RUN_TAG
        hf.attrs['pooling_strategy'] = POOLING_STRATEGY
        hf.attrs['w2v_vector_size'] = W2V_VECTOR_SIZE
        if APPLY_PCA:
            hf.attrs['pca_applied'] = True
            hf.attrs['pca_target_dim'] = PCA_TARGET_DIM

    print("Embeddings saved successfully.")

    gc.collect()
    print(f"\nTotal script execution time: {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
