# ==============================================================================
# MODULE: pipeline/word2vec_embedder.py
# PURPOSE: Handles Word2Vec model training, embedding generation, and pooling.
# VERSION: 2.1 (Corrected DataUtils import)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
import time
from typing import Dict

import h5py
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

from config import Config
# Corrected import for DataUtils
from src.utils.data_utils import DataLoader, DataUtils  # Import DataUtils directly
from src.utils.models_utils import EmbeddingProcessor


class Word2VecEmbedder:
    def __init__(self, config: Config):
        self.config = config
        self.id_map: Dict[str, str] = {}
        DataUtils.print_header("Word2VecEmbedder Initialized")

    def run(self):
        DataUtils.print_header("PIPELINE STEP: Training Word2Vec & Generating Embeddings")
        os.makedirs(str(self.config.WORD2VEC_EMBEDDINGS_DIR), exist_ok=True)

        # This line was causing the error:
        # DataUtils = DataLoader.DataUtils
        # It's removed because DataUtils is now imported directly.

        DataUtils.print_header("Step 1: Loading Protein ID Mapping (if configured)")
        if self.config.ID_MAPPING_MODE != 'none':
            # Assuming DataLoader is instantiated correctly if needed for mapping elsewhere,
            # or that id_map is loaded/passed if this embedder doesn't do mapping itself.
            # For simplicity, if Word2Vec needs its own mapping, it should handle it.
            # If it relies on GCN's mapping, that map needs to be accessible.
            # For now, let's assume it might use a pre-generated map or map IDs later.
            # If Word2Vec needs to *generate* the map, it would need a DataLoader instance.
            # This example assumes id_map is populated if needed by pooling.
            # If no mapping is done by this class, self.id_map remains empty.
            # A more robust solution might involve passing a shared ID map or
            # ensuring each embedder can generate/load its own if necessary.
            # For now, we'll assume the GCN pipeline (if run) populates a map that
            # could be used, or pooling uses original IDs if map is empty.
            # Let's load it similar to how GCN trainer does for consistency:
            if os.path.exists(str(self.config.ID_MAPPING_OUTPUT_FILE)):
                try:
                    mapping_df = pd.read_csv(str(self.config.ID_MAPPING_OUTPUT_FILE), sep='\t', header=None, names=['original', 'mapped'])
                    self.id_map = dict(zip(mapping_df['original'], mapping_df['mapped']))
                    print(f"  Loaded {len(self.id_map)} ID mappings from GCN's output file for Word2Vec.")
                except Exception as e:
                    print(f"  Could not load ID mapping file for Word2Vec: {e}. Using original IDs.")
                    self.id_map = {}
            else:
                print("  ID mapping file not found. Word2Vec will use original FASTA IDs for pooling keys.")
                self.id_map = {}
        else:
            print("  ID mapping mode is 'none'. Word2Vec will use original FASTA IDs for pooling keys.")
            self.id_map = {}

        DataUtils.print_header("Step 2: Preparing FASTA Corpus for Word2Vec")
        fasta_files = []
        if self.config.W2V_INPUT_FASTA_DIR.is_file():
            fasta_files.append(str(self.config.W2V_INPUT_FASTA_DIR))
        elif self.config.W2V_INPUT_FASTA_DIR.is_dir():
            fasta_files = [str(f) for f in self.config.W2V_INPUT_FASTA_DIR.glob('*.fasta')]
        else:
            print(f"ERROR: W2V_INPUT_FASTA_DIR '{self.config.W2V_INPUT_FASTA_DIR}' is not a valid file or directory.")
            return

        if not fasta_files:
            print("ERROR: No FASTA files found for Word2Vec training.")
            return

        print(f"  Found {len(fasta_files)} FASTA file(s) for corpus.")
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
        print(f"  Word2Vec model training finished in {time.time() - model_train_start_time:.2f}s.")
        model_path = str(self.config.WORD2VEC_EMBEDDINGS_DIR / f"word2vec_model_dim{self.config.W2V_VECTOR_SIZE}.model")
        w2v_model.save(model_path)
        print(f"  Word2Vec model saved to: {model_path}")

        DataUtils.print_header("Step 4: Generating Per-Protein Embeddings using Word2Vec")
        protein_embeddings: Dict[str, np.ndarray] = {}
        sequences_for_embedding = []
        for f_path in fasta_files:
            sequences_for_embedding.extend(list(DataLoader.parse_sequences(f_path)))

        if not sequences_for_embedding:
            print("  No sequences found to generate Word2Vec protein embeddings.")
        else:
            for original_id, sequence in tqdm(sequences_for_embedding, desc="  Generating W2V Protein Embeddings", disable=not self.config.DEBUG_VERBOSE):
                residue_vectors = EmbeddingProcessor.get_word2vec_residue_embeddings(sequence, w2v_model, self.config.W2V_VECTOR_SIZE)
                if residue_vectors is not None and residue_vectors.size > 0:
                    protein_vector = EmbeddingProcessor.pool_residue_embeddings(residue_vectors, self.config.W2V_POOLING_STRATEGY, self.config.W2V_VECTOR_SIZE)
                    # Use mapped ID if available, otherwise original ID
                    final_key = self.id_map.get(original_id, original_id)
                    protein_embeddings[final_key] = protein_vector.astype(np.float16)  # Store as float16
                # else:
                # print(f"    Warning: No residue vectors for {original_id}. Skipping.")

        if not protein_embeddings:
            print("  Warning: No protein embeddings generated from Word2Vec.")
        else:
            print(f"  Generated {len(protein_embeddings)} protein embeddings using Word2Vec.")

        output_h5_path = str(self.config.WORD2VEC_EMBEDDINGS_DIR / f"word2vec_dim{self.config.W2V_VECTOR_SIZE}_{self.config.W2V_POOLING_STRATEGY}.h5")
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
                    pca_h5_path = str(self.config.WORD2VEC_EMBEDDINGS_DIR / f"word2vec_dim{self.config.W2V_VECTOR_SIZE}_{self.config.W2V_POOLING_STRATEGY}_pca{pca_dim}.h5")
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
