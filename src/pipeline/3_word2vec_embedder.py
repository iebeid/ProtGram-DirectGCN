# ==============================================================================
# MODULE: pipeline/3_word2vec_embedder.py
# PURPOSE: Trains a Word2Vec model on protein sequences and generates
#          pooled per-protein embeddings.
# ==============================================================================

import os
import glob
import h5py
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
from typing import List, Iterator

# Import from our new project structure
from config import Config
from utils.data_loader import fast_fasta_parser
from utils.embedding_tools import apply_pca


class _FastaCorpus:
    """A memory-efficient corpus for Word2Vec that reads from FASTA files."""

    def __init__(self, fasta_files: List[str]):
        self.fasta_files = fasta_files

    def __iter__(self) -> Iterator[List[str]]:
        for f_path in self.fasta_files:
            for _, sequence in fast_fasta_parser(f_path):
                yield list(sequence)


def _pool_protein_embedding(sequence: str, w2v_model: Word2Vec, pooling_strategy: str) -> np.ndarray:
    """Generates a per-protein embedding by pooling per-residue vectors."""
    embedding_dim = w2v_model.vector_size
    # Use a list to collect vectors of valid residues
    residue_vectors = [w2v_model.wv[residue] for residue in sequence if residue in w2v_model.wv]

    if not residue_vectors:
        return np.zeros(embedding_dim, dtype=np.float32)

    if pooling_strategy == 'mean':
        return np.mean(residue_vectors, axis=0).astype(np.float32)
    elif pooling_strategy == 'sum':
        return np.sum(residue_vectors, axis=0).astype(np.float32)
    elif pooling_strategy == 'max':
        return np.max(residue_vectors, axis=0).astype(np.float32)
    else:
        # Default to mean pooling if strategy is unknown
        return np.mean(residue_vectors, axis=0).astype(np.float32)


# --- Main Orchestration Function for this Module ---
def run_word2vec_training(config: Config):
    """
    The main entry point for the Word2Vec embedding generation step.
    """
    print("\n" + "=" * 80);
    print("### PIPELINE STEP: Training Word2Vec and Generating Embeddings ###");
    print("=" * 80)
    os.makedirs(config.WORD2VEC_EMBEDDINGS_DIR, exist_ok=True)

    # 1. Find FASTA files
    fasta_files = [os.path.normpath(f) for f in glob.glob(os.path.join(config.W2V_INPUT_FASTA_DIR, '*.fasta'))]
    if not fasta_files:
        print(f"Error: No FASTA files found in {config.W2V_INPUT_FASTA_DIR}. Skipping Word2Vec step.");
        return

    # 2. Train Word2Vec model
    print("\nTraining Word2Vec model...")
    corpus = _FastaCorpus(fasta_files)
    w2v_model = Word2Vec(sentences=corpus, vector_size=config.W2V_VECTOR_SIZE, window=config.W2V_WINDOW, min_count=config.W2V_MIN_COUNT, workers=config.W2V_WORKERS, sg=1, epochs=config.W2V_EPOCHS)
    print("Word2Vec model training complete.")

    # 3. Generate and pool embeddings
    print("\nGenerating and pooling per-protein embeddings...")
    pooled_embeddings = {}
    for fasta_file in tqdm(fasta_files, desc="Processing FASTA files for pooling"):
        for protein_id, sequence in fast_fasta_parser(fasta_file):
            pooled_embeddings[protein_id] = _pool_protein_embedding(sequence, w2v_model, config.W2V_POOLING_STRATEGY)
    print(f"Generated {len(pooled_embeddings)} pooled protein embeddings.")

    # 4. Save primary embeddings
    output_filename = f"word2vec_dim{config.W2V_VECTOR_SIZE}_{config.W2V_POOLING_STRATEGY}.h5"
    output_path = os.path.join(config.WORD2VEC_EMBEDDINGS_DIR, output_filename)
    print(f"\nSaving primary Word2Vec embeddings to: {output_path}")
    with h5py.File(output_path, 'w') as hf:
        for protein_id, embedding in pooled_embeddings.items():
            hf.create_dataset(protein_id, data=embedding)

    # 5. Optionally apply and save PCA-reduced embeddings
    if config.APPLY_PCA_TO_W2V:
        pca_embeds = apply_pca(pooled_embeddings, config.PCA_TARGET_DIMENSION, config.RANDOM_STATE)
        if pca_embeds:
            pca_dim = next(iter(pca_embeds.values())).shape[0]
            pca_h5_path = os.path.join(config.WORD2VEC_EMBEDDINGS_DIR, f"word2vec_dim{config.W2V_VECTOR_SIZE}_{config.W2V_POOLING_STRATEGY}_pca{pca_dim}.h5")
            with h5py.File(pca_h5_path, 'w') as hf:
                for key, vector in pca_embeds.items(): hf.create_dataset(key, data=vector)
            print(f"SUCCESS: PCA-reduced Word2Vec embeddings saved to: {pca_h5_path}")

    print("\n### Word2Vec PIPELINE STEP FINISHED ###")
