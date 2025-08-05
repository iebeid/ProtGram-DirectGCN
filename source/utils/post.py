# ==============================================================================
# MODULE: trainers/protgram_helpers.py
# PURPOSE: Contains helper functions for the ProtGramXGCNTrainer.
# VERSION: 1.0 (Created by Gemini Code Assist)
# AUTHOR: Islam Ebeid
# ==============================================================================
import json
from typing import Tuple
import random
from functools import partial
import h5py
from typing import Dict, Optional
from pathlib import Path
import numpy as np
import torch

# Suggested replacement
try:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    TENSORFLOW_AVAILABLE = True
except ImportError:
    # This flag allows the sanity check to be skipped gracefully
    # if TensorFlow is not installed in the environment.
    TENSORFLOW_AVAILABLE = False

from configuration.config import Config
from source.data_builders.graph import DirectedNgramGraph
from source.models.fnn.mlp import MLP
from source.utils.data import DataUtils, FastaUtils, GroundTruthLoader
from source.utils.models import EmbeddingProcessor, EmbeddingLoader


class PostUtils:
    """A class containing helper methods for the ProtGramXGCNTrainer."""

    def __init__(self, config: Config):
        self.config = config

    def pool_lower_level_embeddings(self, graph_obj: DirectedNgramGraph, prev_level_embeddings: np.ndarray,
                                    prev_level_map: Dict[str, int]) -> Optional[torch.Tensor]:
        """
        Pools embeddings from level n-1 to initialize features for level n.

        This is NOT the inverted index method. It's a direct feature engineering step where
        the features for an n-gram (e.g., "ACD") are created by looking up and
        concatenating the final embeddings of its constituent n-1 grams ("AC" and "CD").
        """
        print(f"  Initializing features for n={graph_obj.n_value} by pooling (n-1)-gram constituent embeddings...")
        num_current_nodes = graph_obj.number_of_nodes
        prev_embedding_dim = prev_level_embeddings.shape[1]
        new_features = torch.zeros((num_current_nodes, prev_embedding_dim * 2), dtype=torch.float32)

        # Use the corrected node_names attribute
        for i, current_ngram in enumerate(graph_obj.node_names):
            prefix, suffix = current_ngram[:-1], current_ngram[1:]
            prefix_idx = prev_level_map.get(prefix)
            suffix_idx = prev_level_map.get(suffix)

            prefix_emb = torch.from_numpy(prev_level_embeddings[prefix_idx]) if prefix_idx is not None else torch.zeros(prev_embedding_dim)
            suffix_emb = torch.from_numpy(prev_level_embeddings[suffix_idx]) if suffix_idx is not None else torch.zeros(prev_embedding_dim)

            new_features[i] = torch.cat([prefix_emb, suffix_emb])
        return new_features

    def pool_to_protein_level(self, ngram_embeddings: Dict[int, np.ndarray]) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, Dict[str, float]]]]:
        """Pools the final n-gram embeddings to the protein level."""
        DataUtils.print_header("Step 3: Pooling Final N-gram Embeddings to Protein Level")
        final_n = self.config.GCN_NGRAM_MAX_N
        final_ngram_embeddings = ngram_embeddings.get(final_n)

        if final_ngram_embeddings is None or final_ngram_embeddings.size == 0:
            print(f"  ERROR: No n-gram embeddings found for n={final_n}. Cannot generate protein embeddings.")
            return {}, None

        graph_obj_path = self.config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{final_n}.pkl"
        graph_obj: DirectedNgramGraph = DataUtils.load_object(str(graph_obj_path))
        if graph_obj is None:
            print(f"  ERROR: Failed to load graph object for n={final_n} for pooling.")
            return {}, None
        ngram_map = graph_obj.node_to_idx
        del graph_obj

        protein_sequences = list(FastaUtils.parse_sequences([str(p) for p in self.config.SEQUENCE_FILE_PATHS]))
        pooled_embeddings, attention_weights_by_idx = EmbeddingProcessor.pool_ngram_embeddings_for_protein_fast(
            protein_sequences=protein_sequences, n_val=final_n,
            ngram_map=ngram_map, ngram_embeddings=final_ngram_embeddings,
            strategy=self.config.GCN_PROTEIN_POOLING_STRATEGY
        )

        # Convert attention weight indices to n-gram strings if attention was used
        if attention_weights_by_idx:
            print("  Converting attention weight indices to n-gram strings...")
            idx_to_ngram = {v: k for k, v in ngram_map.items()}
            attention_weights_by_str = {}
            for prot_id, weights_dict in attention_weights_by_idx.items():
                attention_weights_by_str[prot_id] = {
                    idx_to_ngram.get(idx, f"UNKNOWN_IDX_{idx}"): float(weight)
                    for idx, weight in weights_dict.items()
                }
            return pooled_embeddings, attention_weights_by_str

        return pooled_embeddings, None

    def save_final_embeddings(self, final_embeddings_per_model: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, str]:
        """Saves the final generated protein embeddings to H5 files."""
        DataUtils.print_header("Step 4: Saving Generated Embeddings")
        output_paths = {}

        for model_type, protein_embeddings in final_embeddings_per_model.items():
            if protein_embeddings:
                base_filename = f"ProtGram{model_type.capitalize()}_n{self.config.GCN_NGRAM_MAX_N}_embeddings.h5"
                output_path = self.config.RESULTS_GCN_EMBEDDINGS_DIR / base_filename
                DataUtils.write_h5(protein_embeddings, output_path, f"Writing H5 File for {model_type}")
                print(f"\nSUCCESS: Embeddings for '{model_type}' saved to: {output_path}")
                output_paths[f"ProtGram{model_type.capitalize()}"] = str(output_path)
            else:
                print(f"  No embeddings to save for model type '{model_type}'.")
        return output_paths

    def save_attention_weights(self, attention_weights: Dict[str, Dict[str, float]], model_type: str) -> Optional[Path]:
        """
        Saves the protein pooling attention weights to a JSON file.
        """
        if not attention_weights:
            print(f"  No attention weights to save for model type '{model_type}'.")
            return None

        output_dir = self.config.RESULTS_GCN_EMBEDDINGS_DIR
        output_path = output_dir / f"ProtGram{model_type.capitalize()}_n{self.config.GCN_NGRAM_MAX_N}_pooling_attention.json"

        try:
            with open(output_path, 'w') as f:
                json.dump(attention_weights, f, indent=2)
            print(f"SUCCESS: Pooling attention weights for '{model_type}' saved to: {output_path}")
        except Exception as e:
            print(f"ERROR: Could not save attention weights to {output_path}: {e}")
            return None
        return output_path


    def run_sanity_check_ppi(self, embedding_path: str):
        """Performs a quick PPI link prediction task to validate the generated embeddings."""
        DataUtils.print_header("Step 6: Running Sanity Check PPI Task")
        if not TENSORFLOW_AVAILABLE:
            print("  Skipping sanity check: TensorFlow is not installed.")
            return
        if not Path(embedding_path).exists():
            print(f"  Skipping sanity check: Embedding file not found at {embedding_path}")
            return

        sample_size = getattr(self.config, 'GCN_SANITY_CHECK_SAMPLE_SIZE', None)
        pos_pairs = GroundTruthLoader.load_interaction_pairs(str(self.config.POS_INTERACTIONS_PATH), 1)

        if sample_size and len(pos_pairs) > sample_size:
            print(f"  Subsampling positive pairs to {sample_size} for a faster sanity check.")
            pos_pairs = random.sample(pos_pairs, sample_size)

        neg_pairs = GroundTruthLoader.load_interaction_pairs(str(self.config.NEG_INTERACTIONS_PATH), 0, sample_n=len(pos_pairs), random_state=self.config.RANDOM_STATE)
        all_pairs = pos_pairs + neg_pairs
        random.shuffle(all_pairs)
        if not all_pairs:
            print("  Skipping sanity check: No interaction pairs loaded.")
            return

        with EmbeddingLoader(embedding_path) as protein_embeddings:
            pairs_for_eval = [p for p in all_pairs if p[0] in protein_embeddings and p[1] in protein_embeddings]
            print(f"  Found embeddings for {len(pairs_for_eval)} out of {len(all_pairs)} total pairs.")
            if not pairs_for_eval:
                print("  Skipping sanity check: No valid pairs with embeddings found.")
                return

            labels = [p[2] for p in pairs_for_eval]
            try:
                train_pairs, test_pairs = train_test_split(
                    pairs_for_eval, test_size=self.config.GCN_SANITY_CHECK_TEST_SPLIT,
                    random_state=self.config.RANDOM_STATE, stratify=labels
                )
            except ValueError:
                print("  Warning: Stratified split failed for sanity check (likely too few samples in a class). Falling back to a non-stratified split.")
                train_pairs, test_pairs = train_test_split(
                    pairs_for_eval, test_size=self.config.GCN_SANITY_CHECK_TEST_SPLIT, random_state=self.config.RANDOM_STATE)

            first_emb_key = next(iter(protein_embeddings.get_keys()))
            embedding_dim = protein_embeddings[first_emb_key].shape[0]
            edge_feature_dim = embedding_dim * 2

            train_gen = partial(EmbeddingProcessor.generate_edge_features_batched, interaction_pairs=train_pairs, protein_embeddings=protein_embeddings, method='concatenate', batch_size=self.config.EVAL_BATCH_SIZE, embedding_dim=embedding_dim)
            test_gen = partial(EmbeddingProcessor.generate_edge_features_batched, interaction_pairs=test_pairs, protein_embeddings=protein_embeddings, method='concatenate', batch_size=self.config.EVAL_BATCH_SIZE, embedding_dim=embedding_dim)

            output_sig = (tf.TensorSpec(shape=(None, edge_feature_dim), dtype=tf.float16), tf.TensorSpec(shape=(None,), dtype=tf.int32))
            train_ds = tf.data.Dataset.from_generator(train_gen, output_signature=output_sig).prefetch(tf.data.AUTOTUNE)
            test_ds = tf.data.Dataset.from_generator(test_gen, output_signature=output_sig).prefetch(tf.data.AUTOTUNE)

            mlp_params = {'dense1_units': 64, 'dropout1_rate': 0.5, 'dense2_units': 32, 'dropout2_rate': 0.5, 'l2_reg': 1e-5}
            model = MLP(edge_feature_dim, mlp_params, self.config.EVAL_LEARNING_RATE).build()

            print(f"  Training sanity check MLP for {self.config.GCN_SANITY_CHECK_EPOCHS} epochs...")
            model.fit(train_ds, epochs=self.config.GCN_SANITY_CHECK_EPOCHS, verbose=1 if self.config.DEBUG_VERBOSE else 0)

            print("  Evaluating sanity check model...")
            y_true_list, y_pred_list = [], []
            for x_batch, y_batch in test_ds:
                y_true_list.append(y_batch.numpy())
                y_pred_list.append(model.predict_on_batch(x_batch).flatten())

            if not y_true_list:
                print("  Evaluation failed: No data in test set.")
                return

            y_true = np.concatenate(y_true_list)
            y_pred_proba = np.concatenate(y_pred_list)
            y_pred_class = (y_pred_proba > 0.5).astype(int)

            auc = roc_auc_score(y_true, y_pred_proba)
            f1 = f1_score(y_true, y_pred_class)
            precision = precision_score(y_true, y_pred_class)
            recall = recall_score(y_true, y_pred_class)

            print("\n  --- Sanity Check PPI Results ---")
            print(f"  AUC:       {auc:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print("  --------------------------------\n")