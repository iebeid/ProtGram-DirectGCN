# G:/My Drive/Knowledge/Research/TWU/Topics/AI in Proteomics/Protein-protein interaction prediction/Code/ProtDiGCN/src/pipeline/word2vec_embedder.py
# ==============================================================================
# MODULE: pipeline/word2vec_embedder.py
# PURPOSE: Trains a Word2Vec model on protein sequences and generates
#          pooled per-protein embeddings.
# VERSION: 3.0 (Refactored into Word2VecEmbedderPipeline class, _FastaCorpus moved)
# ==============================================================================

import os
import glob
import h5py
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
from typing import List, Iterator # Iterator might not be directly used here anymore

# Import from our new project structure
from src.config import Config
from src.utils.data_utils import DataLoader # Changed from fast_fasta_parser
from src.utils.models_utils import EmbeddingProcessor


class Word2VecEmbedderPipeline:
    """
    Orchestrates the Word2Vec model training and embedding generation pipeline.
    """
    def __init__(self, config: Config):
        """
        Initializes the Word2Vec embedding pipeline.

        Args:
            config (Config): The configuration object for the pipeline.
        """
        self.config = config

    def run_pipeline(self):
        """
        The main entry point for the Word2Vec embedding generation step.
        """
        DataUtils = DataLoader.DataUtils # Access DataUtils via DataLoader if preferred, or import directly
        DataUtils.print_header("PIPELINE STEP: Training Word2Vec and Generating Embeddings")

        os.makedirs(self.config.WORD2VEC_EMBEDDINGS_DIR, exist_ok=True)

        fasta_files = [os.path.normpath(f) for f in glob.glob(os.path.join(str(self.config.W2V_INPUT_FASTA_DIR), '*.fasta')) + \
                       glob.glob(os.path.join(str(self.config.W2V_INPUT_FASTA_DIR), '*.fa'))]
        if not fasta_files:
            print(f"Error: No FASTA files found in {self.config.W2V_INPUT_FASTA_DIR}. Skipping Word2Vec step.")
            return
        print(f"Found {len(fasta_files)} FASTA file(s) to process for Word2Vec.")

        print("\nTraining Word2Vec model...")
        # Use the nested _FastaCorpus from DataLoader
        corpus = DataLoader._FastaCorpus(fasta_files)
        w2v_model = Word2Vec(
            sentences=corpus,
            vector_size=self.config.W2V_VECTOR_SIZE,
            window=self.config.W2V_WINDOW,
            min_count=self.config.W2V_MIN_COUNT,
            workers=self.config.W2V_WORKERS,
            sg=1,  # Use Skip-gram
            epochs=self.config.W2V_EPOCHS
        )
        print("Word2Vec model training complete.")
        embedding_dim = w2v_model.vector_size

        print("\nGenerating and pooling per-protein embeddings...")
        pooled_embeddings = {}
        # For parsing sequences to get protein_id and sequence for pooling
        # We use DataLoader.parse_sequences directly here.
        for fasta_file_path in tqdm(fasta_files, desc="Processing FASTA files for W2V pooling"):
            for protein_id, sequence in DataLoader.parse_sequences(fasta_file_path):
                if not sequence:
                    continue
                # Use EmbeddingProcessor
                residue_embeds = EmbeddingProcessor.get_word2vec_residue_embeddings(
                    sequence, w2v_model, embedding_dim
                )
                # residue_embeds can be an empty array if no valid residues are found,
                # or None if the sequence was empty (though we check `if not sequence` above).
                # The `pool_residue_embeddings` handles empty residue_embeds.
                if residue_embeds is not None:
                    pooled_vec = EmbeddingProcessor.pool_residue_embeddings(
                        residue_embeds, self.config.W2V_POOLING_STRATEGY,
                        embedding_dim_if_empty=embedding_dim
                    )
                    # Ensure pooled_vec is not empty before assignment,
                    # though pool_residue_embeddings with embedding_dim_if_empty should prevent this.
                    if pooled_vec.size > 0:
                        pooled_embeddings[protein_id] = pooled_vec
                    else: # Fallback if pooling somehow results in an empty vector despite dim_if_empty
                        pooled_embeddings[protein_id] = np.zeros(embedding_dim, dtype=np.float32)

                else: # Should ideally not be reached if sequence is non-empty
                    pooled_embeddings[protein_id] = np.zeros(embedding_dim, dtype=np.float32)

        print(f"Generated {len(pooled_embeddings)} pooled protein embeddings.")

        output_filename = f"word2vec_dim{self.config.W2V_VECTOR_SIZE}_{self.config.W2V_POOLING_STRATEGY}_full_dim.h5"
        output_path = os.path.join(str(self.config.WORD2VEC_EMBEDDINGS_DIR), output_filename)
        print(f"\nSaving primary Word2Vec embeddings to: {output_path}")
        with h5py.File(output_path, 'w') as hf:
            for protein_id, embedding in pooled_embeddings.items():
                if embedding is not None and embedding.size > 0:
                    hf.create_dataset(protein_id, data=embedding)

        if self.config.APPLY_PCA_TO_W2V and len(pooled_embeddings) > 1:
            # Use EmbeddingProcessor for PCA
            pca_embeds = EmbeddingProcessor.apply_pca(
                pooled_embeddings, self.config.PCA_TARGET_DIMENSION, self.config.RANDOM_STATE
            )
            if pca_embeds:
                first_valid_pca_emb = next((v for v in pca_embeds.values() if v is not None and v.size > 0), None)
                if first_valid_pca_emb is not None:
                    pca_dim = first_valid_pca_emb.shape[0]
                    pca_h5_path = os.path.join(str(self.config.WORD2VEC_EMBEDDINGS_DIR),
                                               f"word2vec_dim{self.config.W2V_VECTOR_SIZE}_{self.config.W2V_POOLING_STRATEGY}_pca{pca_dim}.h5")
                    with h5py.File(pca_h5_path, 'w') as hf:
                        for key, vector in pca_embeds.items():
                            if vector is not None and vector.size > 0:
                                hf.create_dataset(key, data=vector)
                    print(f"SUCCESS: PCA-reduced Word2Vec embeddings saved to: {pca_h5_path}")
                else:
                    print("Word2Vec PCA: No valid PCA embeddings to determine dimension for saving.")
            else:
                print("Word2Vec PCA: PCA application did not return embeddings.")

        DataUtils.print_header("Word2Vec PIPELINE STEP FINISHED")


