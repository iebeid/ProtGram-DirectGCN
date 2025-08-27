# ==============================================================================
# MODULE: trainers/word2vec.py
# PURPOSE: Handles Word2Vec model training, embedding generation, and pooling.
# VERSION: 3.0 (Integrated consistent ID mapping and centralized data sources)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
import time # noqa
from typing import Dict, Optional, List

import numpy as np
from gensim.models import Word2Vec
from tqdm.auto import tqdm

from configuration.config import Config
from source.utils.data.data_utils import DataUtils
from source.utils.data.fasta_utils import FastaUtils
from source.utils.fs.file_utils import FileUtils
from source.utils.post.embedding_processor import EmbeddingProcessor


class Word2VecEmbedder:
    def __init__(self, config: Config):
        self.config = config
        DataUtils.print_header("Word2VecEmbedder Initialized")

    def run(self) -> Optional[Dict[str, str]]:
        """
        Main entry point for the Word2Vec pipeline.
        """
        DataUtils.print_header("PIPELINE STEP: Training Word2Vec & Generating Embeddings")
        self.config.RESULTS_W2V_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        fasta_paths = self.config.SEQUENCE_FILE_PATHS
        if not fasta_paths:
            print("ERROR: No FASTA files configured for Word2Vec.")
            return None

        fasta_files = [str(p) for p in fasta_paths]
        corpus = FastaUtils.FastaCorpus(fasta_files)

        w2v_model = self._train_w2v_model(corpus)
        protein_embeddings = self._generate_protein_embeddings(w2v_model, fasta_files)

        # --- REFACTOR: Use a dedicated helper to save embeddings and handle PCA ---
        # This makes the logic consistent with the ProtGram-XGCN trainer.
        output_paths = self._save_embeddings_and_apply_pca(protein_embeddings)

        del w2v_model, corpus, protein_embeddings
        gc.collect()

        DataUtils.print_header("Word2Vec Embedding PIPELINE STEP FINISHED")
        return output_paths

    def _train_w2v_model(self, corpus: FastaUtils.FastaCorpus) -> Word2Vec:
        """Trains the Word2Vec model on the provided corpus."""
        DataUtils.print_header("Step 2: Training Word2Vec Model")
        print(
            f"  Training Word2Vec model (vector_size={self.config.W2V_VECTOR_SIZE}, window={self.config.W2V_WINDOW}, epochs={self.config.W2V_EPOCHS})...")
        model_train_start_time = time.time()
        w2v_model = Word2Vec(
            corpus, vector_size=self.config.W2V_VECTOR_SIZE, window=self.config.W2V_WINDOW,
            min_count=self.config.W2V_MIN_COUNT, epochs=self.config.W2V_EPOCHS,
            workers=self.config.W2V_WORKERS, sg=1, hs=0, negative=5, seed=self.config.RANDOM_STATE
        )
        print(f"  Word2Vec model training finished in {time.time() - model_train_start_time:.2f}s.")
        return w2v_model

    def _generate_protein_embeddings(self, w2v_model: Word2Vec, fasta_files: List[str]) -> Dict[str, np.ndarray]:
        """Generates per-protein embeddings using the trained Word2Vec model."""
        DataUtils.print_header("Step 3: Generating Per-Protein Embeddings using Word2Vec")
        protein_embeddings: Dict[str, np.ndarray] = {}
        # Use the generator directly to avoid loading all sequences into memory
        sequences_for_embedding = FastaUtils.parse_sequences(
            fasta_files,
            perform_cleaning=self.config.PROTGRAM_CLEAN_FASTA_ON_PARSE,
            min_len=self.config.PROTGRAM_FASTA_MIN_LEN,
            max_len=self.config.PROTGRAM_FASTA_MAX_LEN,
            alphabet_type=self.config.PROTGRAM_FASTA_ALPHABET
        )

        for original_id, sequence in tqdm(sequences_for_embedding, desc="  Generating W2V Protein Embeddings"):
            residue_vectors = EmbeddingProcessor.get_word2vec_residue_embeddings(sequence, w2v_model,
                                                                                 self.config.W2V_VECTOR_SIZE)
            if residue_vectors is not None and residue_vectors.size > 0:
                protein_vector = EmbeddingProcessor.pool_residue_embeddings(residue_vectors,
                                                                            self.config.W2V_POOLING_STRATEGY,
                                                                            self.config.W2V_VECTOR_SIZE)
                protein_embeddings[original_id] = protein_vector

        if not protein_embeddings:
            print("  Warning: No protein embeddings generated from Word2Vec.")

        return protein_embeddings

    def _save_embeddings_and_apply_pca(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, str]:
        """Saves final protein embeddings and their PCA versions to H5 files."""
        output_paths = {}
        if not embeddings:
            print("  No embeddings generated for Word2Vec. Skipping save.")
            return output_paths

        model_name = "Word2Vec-Generated"
        output_dir = self.config.RESULTS_W2V_EMBEDDINGS_DIR
        dim = self.config.W2V_VECTOR_SIZE
        pooling = self.config.W2V_POOLING_STRATEGY

        output_path = output_dir / f"word2vec_dim{dim}_{pooling}.h5"
        FileUtils.write_h5(embeddings, output_path, f"Writing H5 for {model_name}")
        output_paths[model_name] = str(output_path)

        if self.config.APPLY_PCA_TO_W2V and self.config.PCA_TARGET_DIMENSION > 0:
            pca_path = EmbeddingProcessor.apply_pca_to_h5(
                input_h5_path=output_path,
                output_dir=output_dir,
                target_dimension=self.config.PCA_TARGET_DIMENSION,
                random_seed=self.config.RANDOM_STATE
            )
            if str(pca_path) != str(output_path):
                output_paths[f"{model_name}_pca"] = str(pca_path)
        return output_paths