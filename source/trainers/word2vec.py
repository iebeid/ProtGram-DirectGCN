# ==============================================================================
# MODULE: trainers/word2vec.py
# PURPOSE: Handles Word2Vec model trainers, embedding generation, and pooling.
# VERSION: 2.1 (Corrected DataUtils import)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
import time
from typing import Dict, Optional, Union, Mapping

import h5py
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

from configuration.config import Config
# Corrected import for DataUtils
from source.utils.data import DataLoader, DataUtils, IDMapper  # Import DataUtils and the new IDMapper
from source.utils.models import EmbeddingProcessor


class Word2VecEmbedder:
    def __init__(self, config: Config):
        self.config = config
        self.id_mapper: Optional[Union[Dict[str, str], IDMapper]] = None
        DataUtils.print_header("Word2VecEmbedder Initialized")

    def run(self):
        DataUtils.print_header("PIPELINE STEP: Training Word2Vec & Generating Embeddings")
        os.makedirs(str(self.config.RESULTS_W2V_EMBEDDINGS_DIR), exist_ok=True)

        # This line was causing the error:
        # DataUtils = DataLoader.DataUtils
        # It's removed because DataUtils is now imported directly.

        DataUtils.print_header("Step 1: Loading Protein ID Mapping (if configured)")
        # FIX: Use the robust DataLoader to handle ID mapping safely.
        data_loader = DataLoader(config=self.config)
        self.id_mapper = data_loader.generate_id_maps()

        # Now, we need a 'with' block if the mapper is the on-disk SQLite version
        if isinstance(self.id_mapper, IDMapper):
            with self.id_mapper as mapper:
                # The main logic is now inside the 'with' block
                return self._execute_embedding_logic(mapper)
        else:
            # If it's a simple dictionary (from regex/api) or None, run directly.
            return self._execute_embedding_logic(self.id_mapper)

    def _execute_embedding_logic(self, id_mapper: Optional[Mapping]):
        """The core logic for Word2Vec, now accepting a mapper object."""

        DataUtils.print_header("Step 2: Preparing FASTA Corpus for Word2Vec")
        # FIX: The Word2Vec embedder should use the same centrally-defined sequence files
        # as the rest of the pipeline, which are located in config.SEQUENCE_FILE_PATHS.
        fasta_paths = self.config.SEQUENCE_FILE_PATHS
        if not fasta_paths:
            print("ERROR: No FASTA files configured in 'config.SEQUENCE_FILE_PATHS' for Word2Vec.")
            return

        # Convert Path objects to strings for gensim compatibility
        fasta_files = [str(p) for p in fasta_paths]
        print(f"  Found {len(fasta_files)} FASTA file(s) for corpus: {[os.path.basename(f) for f in fasta_files]}")
        corpus = DataLoader._FastaCorpus(fasta_files)  # Use the nested _FastaCorpus

        DataUtils.print_header("Step 3: Training Word2Vec Model")
        print(f"  Training Word2Vec model (vector_size={self.config.W2V_VECTOR_SIZE}, window={self.config.W2V_WINDOW}, epochs={self.config.W2V_EPOCHS}, workers={self.config.W2V_WORKERS})...")
        model_train_start_time = time.time()
        w2v_model = Word2Vec(
            corpus,
            vector_size=self.config.W2V_VECTOR_SIZE,
            window=self.config.W2V_WINDOW,
            min_count=self.config.W2V_MIN_COUNT,
            epochs=self.config.W2V_EPOCHS,
            workers=self.config.W2V_WORKERS,
            sg=1,  # Skip-gram
            hs=0,  # Negative sampling
            negative=5,  # Number of negative samples
            seed=self.config.RANDOM_STATE
        )
        print(f"  Word2Vec model trainers finished in {time.time() - model_train_start_time:.2f}s.")
        model_path = str(self.config.RESULTS_W2V_EMBEDDINGS_DIR / f"word2vec_model_dim{self.config.W2V_VECTOR_SIZE}.model")
        w2v_model.save(model_path)
        print(f"  Word2Vec model saved to: {model_path}")

        DataUtils.print_header("Step 4: Generating Per-Protein Embeddings using Word2Vec")
        protein_embeddings: Dict[str, np.ndarray] = {}
        # FIX: The parse_sequences function takes a list of file paths.
        sequences_for_embedding = list(DataLoader.parse_sequences(fasta_files))

        if not sequences_for_embedding:
            print("  No sequences found to generate Word2Vec protein embeddings.")
        else:
            for original_id, sequence in tqdm(sequences_for_embedding, desc="  Generating W2V Protein Embeddings", disable=not self.config.DEBUG_VERBOSE):
                residue_vectors = EmbeddingProcessor.get_word2vec_residue_embeddings(sequence, w2v_model, self.config.W2V_VECTOR_SIZE)
                if residue_vectors is not None and residue_vectors.size > 0:
                    protein_vector = EmbeddingProcessor.pool_residue_embeddings(residue_vectors, self.config.W2V_POOLING_STRATEGY, self.config.W2V_VECTOR_SIZE)
                    # Use mapped ID if available, otherwise original ID
                    final_key = id_mapper.get(original_id, original_id) if id_mapper else original_id
                    protein_embeddings[final_key] = protein_vector.astype(np.float16)  # Store as float16
                # else:
                # print(f"    Warning: No residue vectors for {original_id}. Skipping.")

        if not protein_embeddings:
            print("  Warning: No protein embeddings generated from Word2Vec.")
        else:
            print(f"  Generated {len(protein_embeddings)} protein embeddings using Word2Vec.")

        output_h5_path = str(self.config.RESULTS_W2V_EMBEDDINGS_DIR / f"word2vec_dim{self.config.W2V_VECTOR_SIZE}_{self.config.W2V_POOLING_STRATEGY}.h5")
        with h5py.File(output_h5_path, 'w') as hf:
            for key, vector in tqdm(protein_embeddings.items(), desc="  Writing H5 File", disable=not self.config.DEBUG_VERBOSE):
                if vector is not None and vector.size > 0:
                    hf.create_dataset(key, data=vector)  # Already float16
        print(f"\nSUCCESS: Word2Vec embeddings saved to: {output_h5_path}")

        if self.config.APPLY_PCA_TO_W2V and protein_embeddings:
            DataUtils.print_header("Step 5: Applying PCA to Word2Vec Embeddings")
            # apply_pca expects float32 input for stability, but can output float16
            pca_embeds = EmbeddingProcessor.apply_pca(protein_embeddings, self.config.PCA_TARGET_DIMENSION, self.config.RANDOM_STATE, output_dtype=np.float16)
            if pca_embeds:
                first_valid_pca_emb = next((v for v in pca_embeds.values() if v is not None and v.size > 0), None)
                if first_valid_pca_emb is not None:
                    pca_dim = first_valid_pca_emb.shape[0]
                    pca_h5_path = str(self.config.RESULTS_W2V_EMBEDDINGS_DIR / f"word2vec_dim{self.config.W2V_VECTOR_SIZE}_{self.config.W2V_POOLING_STRATEGY}_pca{pca_dim}.h5")
                    with h5py.File(pca_h5_path, 'w') as hf:
                        for key, vector in tqdm(pca_embeds.items(), desc="  Writing PCA H5 File", disable=not self.config.DEBUG_VERBOSE):
                            if vector is not None and vector.size > 0:
                                hf.create_dataset(key, data=vector)  # Already float16
                    print(f"  SUCCESS: PCA-reduced Word2Vec embeddings saved to: {pca_h5_path}")
                else:
                    print("  PCA Warning: No valid PCA embeddings to determine dimension for saving.")

            elif protein_embeddings:
                print("  Warning: PCA was requested for Word2Vec but resulted in no embeddings.")

        del w2v_model, corpus, protein_embeddings
        if 'pca_embeds' in locals(): del pca_embeds
        gc.collect()
        DataUtils.print_header("Word2Vec Embedding PIPELINE STEP FINISHED")
