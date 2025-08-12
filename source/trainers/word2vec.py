# ==============================================================================
# MODULE: trainers/word2vec.py
# PURPOSE: Handles Word2Vec model training, embedding generation, and pooling.
# VERSION: 3.0 (Integrated consistent ID mapping and centralized data sources)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
import time
from contextlib import nullcontext
from typing import Dict, Optional, Union, Mapping

import h5py
import numpy as np
from gensim.models import Word2Vec
from tqdm.auto import tqdm

from configuration.config import Config
from source.utils.data import FastaUtils, DataUtils, IDMapGenerator
from source.utils.post import EmbeddingProcessor


class Word2VecEmbedder:
    def __init__(self, config: Config):
        self.config = config
        DataUtils.print_header("Word2VecEmbedder Initialized")

    def run(self) -> Optional[str]:
        """
        Main entry point for the Word2Vec pipeline. It handles ID mapping
        and then executes the core embedding generation logic.
        """
        DataUtils.print_header("PIPELINE STEP: Training Word2Vec & Generating Embeddings")
        self.config.RESULTS_W2V_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        id_mapper = self._load_id_map()

        # Use a context manager if the mapper is the on-disk SQLite version
        context = id_mapper if isinstance(id_mapper, IDMapGenerator) else nullcontext(id_mapper)
        with context as mapper:
            return self._execute_embedding_logic(mapper)

    def _load_id_map(self) -> Optional[Mapping]:
        """Loads the UniProt ID mapping file if configured."""
        if getattr(self.config, 'ID_MAPPING_MODE', 'none') != 'none':
            print("  Loading Protein ID Mapping for consistency...")
            id_mapper_instance = IDMapGenerator(config=self.config)
            id_map_result = id_mapper_instance.generate_id_maps()
            print(f"  ID mapping result of type '{type(id_map_result)}' loaded.")
            return id_map_result
        return None

    def _execute_embedding_logic(self, id_mapper: Optional[Mapping]) -> Optional[str]:
        """The core logic for Word2Vec, now accepting a mapper object."""
        DataUtils.print_header("Step 1: Preparing FASTA Corpus for Word2Vec")
        fasta_paths = self.config.SEQUENCE_FILE_PATHS
        if not fasta_paths:
            print("ERROR: No FASTA files configured in 'config.SEQUENCE_FILE_PATHS' for Word2Vec.")
            return None

        fasta_files = [str(p) for p in fasta_paths]
        print(f"  Found {len(fasta_files)} FASTA file(s) for corpus: {[os.path.basename(f) for f in fasta_files]}")
        corpus = FastaUtils.FastaCorpus(fasta_files)

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

        DataUtils.print_header("Step 3: Generating Per-Protein Embeddings using Word2Vec")
        protein_embeddings: Dict[str, np.ndarray] = {}
        sequences_for_embedding = list(FastaUtils.parse_sequences(fasta_files))

        for original_id, sequence in tqdm(sequences_for_embedding, desc="  Generating W2V Protein Embeddings"):
            residue_vectors = EmbeddingProcessor.get_word2vec_residue_embeddings(sequence, w2v_model,
                                                                                 self.config.W2V_VECTOR_SIZE)
            if residue_vectors is not None and residue_vectors.size > 0:
                protein_vector = EmbeddingProcessor.pool_residue_embeddings(residue_vectors,
                                                                            self.config.W2V_POOLING_STRATEGY,
                                                                            self.config.W2V_VECTOR_SIZE)
                final_key = id_mapper.get(original_id, original_id) if id_mapper else original_id
                protein_embeddings[final_key] = protein_vector

        if not protein_embeddings:
            print("  Warning: No protein embeddings generated from Word2Vec.")
            return None

        print(f"  Generated {len(protein_embeddings)} protein embeddings using Word2Vec.")

        output_h5_path = self.config.RESULTS_W2V_EMBEDDINGS_DIR / f"word2vec_dim{self.config.W2V_VECTOR_SIZE}_{self.config.W2V_POOLING_STRATEGY}.h5"
        DataUtils.write_h5(protein_embeddings, output_h5_path, "Writing Word2Vec H5 File")

        # --- NEW: Apply PCA for consistency with other embedding pipelines ---
        final_path = EmbeddingProcessor.apply_pca_to_h5(
            input_h5_path=output_h5_path,
            output_dir=self.config.RESULTS_W2V_EMBEDDINGS_DIR,
            target_dimension=self.config.PCA_TARGET_DIMENSION,
            random_seed=self.config.RANDOM_STATE
        )

        print(f"\nSUCCESS: Word2Vec embeddings processing complete. Final file: {final_path}")

        del w2v_model, corpus, protein_embeddings
        gc.collect()
        DataUtils.print_header("Word2Vec Embedding PIPELINE STEP FINISHED")
        return str(final_path)