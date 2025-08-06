# ==============================================================================
# MODULE: utils/models.py
# PURPOSE: Contains tools for loading and post-processing embeddings, such as PCA,
#          normalization, pooling, and edge feature creation.
# VERSION: 6.2 (Final fix for embedding extraction for all GNN types)
# AUTHOR: Islam Ebeid
# ==============================================================================

from pathlib import Path
from typing import Dict, Optional, List, Tuple, Set, Union, TYPE_CHECKING, Iterator

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm.auto import tqdm

# Local imports are safe here as this is a core utility module
from .data import DataUtils

if TYPE_CHECKING:
    from gensim.models import Word2Vec
    from configuration.config import Config
    from source.data_builders.graph import DirectedNgramGraph


class EarlyStopper:
    """A simple early stopper to monitor loss and stop training when it stops improving."""

    def __init__(self, patience: int = 1, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class EmbeddingLoader:
    """
    A lazy loader for HDF5 embeddings that acts like a dictionary.
    It keeps the H5 file open and retrieves embeddings on-the-fly, which is
    highly memory-efficient. It should be used as a context manager.
    """

    def __init__(self, h5_path: Union[str, Path]):
        self.h5_path = Path(h5_path)
        self._h5_file: Optional[h5py.File] = None
        self._keys: Optional[Set[str]] = None

    def __enter__(self) -> 'EmbeddingLoader':
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {self.h5_path}")
        try:
            self._h5_file = h5py.File(self.h5_path, 'r')
            self._keys = set(self._h5_file.keys())
            print(f"Opened H5 file: {self.h5_path.name}, found {len(self._keys)} keys.")
        except Exception as e:
            if self._h5_file:
                self._h5_file.close()
            raise IOError(f"Could not open or read HDF5 file {self.h5_path}: {e}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._h5_file:
            self._h5_file.close()
            print(f"Closed H5 file: {self.h5_path.name}")
            self._h5_file = None
            self._keys = None

    def __contains__(self, key: str) -> bool:
        if self._keys is None:
            raise RuntimeError("EmbeddingLoader used outside of context or after exit.")
        return key in self._keys

    def __getitem__(self, key: str) -> np.ndarray:
        if self._h5_file is None or self._keys is None:
            raise RuntimeError("EmbeddingLoader used outside of context or after exit.")
        if key in self._keys:
            return self._h5_file[key][:].astype(np.float16)
        raise KeyError(f"Key '{key}' not found in {self.h5_path}")

    def __len__(self) -> int:
        return len(self._keys) if self._keys is not None else 0

    def get_keys(self) -> Set[str]:
        if self._keys is None:
            raise RuntimeError("EmbeddingLoader used outside of context or after exit.")
        return set(self._keys)


class EmbeddingProcessor:
    """
    A class for handling common processing tasks for embeddings.
    """

    @staticmethod
    def apply_pca(
            embeddings_dict: Dict[str, np.ndarray], target_dim: int, random_seed: int,
            output_dtype: np.dtype = np.float16) -> Optional[Dict[str, np.ndarray]]:
        """
        Applies PCA to an in-memory dictionary of embeddings.
        """
        if not embeddings_dict:
            print("PCA Error: No embeddings provided to transform.")
            return None

        # FIX: Explicitly filter out embeddings containing NaN values to prevent PCA from failing.
        filtered_embeddings_list = []
        filtered_ids = []
        for k, v in embeddings_dict.items():
            if v is not None and v.size > 0 and not np.isnan(v).any():
                filtered_embeddings_list.append(v.astype(np.float32))
                filtered_ids.append(k)
            elif v is not None and np.isnan(v).any():
                print(f"    PCA Warning: Skipping protein '{k}' because its embedding contains NaN values.")

        if not filtered_embeddings_list:
            print("PCA Error: All embedding vectors are None or empty. Skipping PCA.")
            return None

        print(f"\nApplying PCA to reduce dimensions to {target_dim} for {len(filtered_ids)} embeddings...")
        embedding_matrix = np.array(filtered_embeddings_list, dtype=np.float32)
        original_dimension = embedding_matrix.shape[1]
        n_samples = embedding_matrix.shape[0]

        actual_target_dim = min(target_dim, original_dimension, n_samples)
        if actual_target_dim < target_dim:
            print(f"PCA Warning: Adjusted target dimension from {target_dim} to {actual_target_dim} due to data constraints.")
        if actual_target_dim <= 0:
            print(f"PCA Error: Cannot perform PCA with target dimension {actual_target_dim}. Skipping.")
            return {pid: emb.astype(output_dtype) for pid, emb in zip(filtered_ids, filtered_embeddings_list)}

        try:
            scaler = StandardScaler()
            scaled_embeddings = scaler.fit_transform(embedding_matrix)
            pca = PCA(n_components=actual_target_dim, random_state=random_seed)
            transformed_embeddings = pca.fit_transform(scaled_embeddings)
            print(f"PCA Applied: Original shape: {embedding_matrix.shape}, Transformed shape: {transformed_embeddings.shape}")
            if pca.explained_variance_ratio_ is not None:
                print(f"Explained variance by {actual_target_dim} components: {np.sum(pca.explained_variance_ratio_):.4f}")
            return {pid: transformed_vec.astype(output_dtype) for pid, transformed_vec in zip(filtered_ids, transformed_embeddings)}
        except Exception as e:
            print(f"PCA Error during transformation: {e}. Skipping PCA.")
            return {pid: emb.astype(output_dtype) for pid, emb in zip(filtered_ids, filtered_embeddings_list)}

    @staticmethod
    def apply_pca_to_h5(input_h5_path: Path, output_dir: Path, target_dimension: int, random_seed: int) -> Path:
        """
        Reads embeddings from an HDF5 file, applies PCA, and saves the
        transformed embeddings to a new HDF5 file in the specified output directory.
        This is the refactored logic from the old `PostUtils` class.

        Args:
            input_h5_path (Path): Path to the input HDF5 file.
            output_dir (Path): Directory to save the new HDF5 file.
            target_dimension (int): The desired dimension after PCA.
            random_seed (int): The random seed for PCA reproducibility.

        Returns:
            Path: The path to the newly created HDF5 file, or the original path on failure.
        """
        print(f"Applying PCA to file: '{input_h5_path.name}'. Target dimension: {target_dimension}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_h5_path = output_dir / f"{input_h5_path.stem}_pca{target_dimension}.h5"

        if output_h5_path.exists():
            print(f"  PCA-processed file already exists: {output_h5_path.name}. Skipping.")
            return output_h5_path

        try:
            # Step 1: Load all embeddings from the file into a dictionary using the robust loader
            with EmbeddingLoader(input_h5_path) as loader:
                # Load all into memory for global PCA
                embeddings_dict = {key: loader[key] for key in loader.get_keys()}

            if not embeddings_dict:
                print(f"  Warning: No embeddings found in '{input_h5_path.name}'. Cannot apply PCA.")
                return input_h5_path  # Return original path

            # Step 2: Apply PCA to the in-memory dictionary using the existing robust method
            transformed_embeddings = EmbeddingProcessor.apply_pca(
                embeddings_dict, target_dimension, random_seed
            )

            # Step 3: Save the transformed embeddings to the new file
            if transformed_embeddings:
                DataUtils.write_h5(transformed_embeddings, output_h5_path, f"Writing PCA H5 for {input_h5_path.name}")
                print(f"  Successfully saved PCA-transformed embeddings to '{output_h5_path.name}'")
                return output_h5_path
            else:
                print(f"  PCA transformation returned no results. Returning original path.")
                return input_h5_path

        except Exception as e:
            print(f"  ERROR during file-based PCA processing for '{input_h5_path.name}': {e}")
            return input_h5_path  # Return original path on failure

    @staticmethod
    def l2_normalize_torch(embeddings: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        norm = torch.norm(embeddings, p=2, dim=-1, keepdim=True)
        return embeddings / (norm + eps)

    @staticmethod
    def extract_transformer_residue_embeddings(raw_model_output: np.ndarray, original_sequence_length: int,
                                               is_t5_model: bool) -> np.ndarray:
        if original_sequence_length <= 0 or raw_model_output.shape[0] == 0:
            return np.array([], dtype=raw_model_output.dtype).reshape(0, raw_model_output.shape[-1])
        if is_t5_model:
            return raw_model_output[:min(raw_model_output.shape[0], original_sequence_length), :]
        else:
            # For BERT-like models, skip the [CLS] token at the beginning
            return raw_model_output[1:min(raw_model_output.shape[0], original_sequence_length + 1), :]

    @staticmethod
    def get_word2vec_residue_embeddings(sequence: str, w2v_model: 'Word2Vec',
                                        embedding_dim: int) -> Optional[np.ndarray]:
        if not sequence: return np.zeros((0, embedding_dim), dtype=np.float32)
        if not hasattr(w2v_model, 'wv'): return np.zeros((0, embedding_dim), dtype=np.float32)
        residue_vectors = [w2v_model.wv[residue] for residue in sequence if residue in w2v_model.wv]
        return np.array(residue_vectors, dtype=np.float32) if residue_vectors else np.zeros((0, embedding_dim), dtype=np.float32)

    @staticmethod
    def pool_residue_embeddings(residue_embeddings: np.ndarray, strategy: str,
                                embedding_dim_if_empty: Optional[int] = None) -> np.ndarray:
        if residue_embeddings is None or residue_embeddings.shape[0] == 0:
            return np.zeros(embedding_dim_if_empty, dtype=np.float32) if embedding_dim_if_empty else np.array([], dtype=np.float32)
        if strategy == 'mean': return np.mean(residue_embeddings, axis=0)
        if strategy == 'sum': return np.sum(residue_embeddings, axis=0)
        if strategy == 'max': return np.max(residue_embeddings, axis=0)
        if strategy == 'attention':
            # A simple, non-parametric attention mechanism.
            # The score for each residue is its dot product with the mean embedding.
            if residue_embeddings.shape[0] > 1:
                mean_vec = np.mean(residue_embeddings, axis=0, keepdims=True)
                attention_scores = np.dot(residue_embeddings, mean_vec.T).flatten()
                attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores))
                return np.dot(attention_weights, residue_embeddings)
            return np.mean(residue_embeddings, axis=0) # Fallback for single-residue sequences
        return np.mean(residue_embeddings, axis=0)

    @staticmethod
    def pool_ngram_embeddings_for_protein_fast(protein_sequences: List[Tuple[str, str]], n_val: int,
                                               ngram_map: Dict[str, int], ngram_embeddings: np.ndarray,
                                               strategy: str = 'mean') -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[int, float]]]:
        """
        A method for pooling n-gram embeddings to the protein level.
        Supports 'mean', 'sum', and 'max' via an efficient inverted index.
        Supports 'attention' via a protein-by-protein iteration.

        Returns:
            A tuple containing (pooled_embeddings, attention_weights).
            The attention_weights dict is {protein_id: {ngram_idx: weight}} and is
            only populated if strategy is 'attention'.
        """
        print(f"  Starting protein-level pooling (Strategy: {strategy})...")
        if not protein_sequences: return {}

        # --- Strategy Dispatcher ---
        if strategy in ['mean', 'sum', 'max']:
            # --- Fast Inverted Index Method ---
            print("    Using fast inverted index method.")
            num_proteins = len(protein_sequences)
            embedding_dim = ngram_embeddings.shape[1]
            protein_ids_ordered = [p_data[0] for p_data in protein_sequences]

            if strategy == 'max':
                protein_pooled_values = np.full((num_proteins, embedding_dim), -np.inf, dtype=np.float32)
            else:  # for 'mean' and 'sum'
                protein_pooled_values = np.zeros((num_proteins, embedding_dim), dtype=np.float32)
            protein_ngram_counts = np.zeros(num_proteins, dtype=np.int32)

            ngram_idx_to_protein_indices = [[] for _ in range(len(ngram_embeddings))]
            for prot_idx, (_, seq) in enumerate(tqdm(protein_sequences, desc="    Building inverted index", leave=False)):
                if len(seq) >= n_val:
                    for i in range(len(seq) - n_val + 1):
                        ngram_idx = ngram_map.get("".join(seq[i:i + n_val]))
                        if ngram_idx is not None:
                            ngram_idx_to_protein_indices[ngram_idx].append(prot_idx)

            for ngram_idx, prot_indices in enumerate(tqdm(ngram_idx_to_protein_indices, desc="    Aggregating embeddings", leave=False)):
                if prot_indices:
                    ngram_emb = ngram_embeddings[ngram_idx].astype(np.float32)
                    if strategy == 'max':
                        np.maximum(protein_pooled_values[prot_indices], ngram_emb, out=protein_pooled_values[prot_indices])
                    else:  # 'mean' or 'sum'
                        protein_pooled_values[prot_indices] += ngram_emb
                    protein_ngram_counts[prot_indices] += 1

            valid_counts_mask = protein_ngram_counts > 0
            if strategy == 'mean':
                protein_pooled_values[valid_counts_mask] /= protein_ngram_counts[valid_counts_mask, np.newaxis]
            elif strategy == 'max':
                protein_pooled_values[~valid_counts_mask] = 0

            pooled_embeddings = {protein_ids_ordered[i]: protein_pooled_values[i].astype(ngram_embeddings.dtype)
                                 for i in range(num_proteins) if valid_counts_mask[i]}
            return pooled_embeddings, {}

        elif strategy == 'attention':
            # --- HYBRID METHOD for Attention ---
            # We use an index to speed up the process, separating the slow n-gram lookup
            # from the actual attention calculation.
            print("    Using hybrid method for attention pooling (pre-indexing n-grams).")

            pooled_embeddings = {}
            attention_weights_log = {}

            # Step 1: Pre-build the protein -> [n-gram indices] mapping.
            num_proteins = len(protein_sequences)
            protein_ids_ordered = [p_data[0] for p_data in protein_sequences]
            protein_to_ngram_indices = [[] for _ in range(num_proteins)]
            for prot_idx, (_, seq) in enumerate(tqdm(protein_sequences, desc="    Building protein->n-gram index")):
                if len(seq) >= n_val:
                    for i in range(len(seq) - n_val + 1):
                        ngram_idx = ngram_map.get("".join(seq[i:i + n_val]))
                        if ngram_idx is not None:
                            protein_to_ngram_indices[prot_idx].append(ngram_idx)

            # Step 2: Iterate through the pre-built index to calculate attention.
            for prot_idx, ngram_indices in enumerate(tqdm(protein_to_ngram_indices, desc="    Calculating Attention")):
                if not ngram_indices: continue

                prot_id = protein_ids_ordered[prot_idx]
                protein_ngrams_arr = ngram_embeddings[ngram_indices].astype(np.float32)

                if protein_ngrams_arr.shape[0] > 1:
                    mean_vec = np.mean(protein_ngrams_arr, axis=0, keepdims=True)
                    attention_scores = np.dot(protein_ngrams_arr, mean_vec.T).flatten()
                    exp_scores = np.exp(attention_scores - np.max(attention_scores))
                    attention_weights = exp_scores / np.sum(exp_scores)
                    pooled_emb = np.dot(attention_weights, protein_ngrams_arr)
                    # Log the weights by their n-gram index
                    attention_weights_log[prot_id] = {
                        idx: weight for idx, weight in zip(ngram_indices, attention_weights)
                    }
                elif protein_ngrams_arr.shape[0] == 1:
                    pooled_emb = protein_ngrams_arr[0]
                    # Attention for a single item is 1.0
                    attention_weights_log[prot_id] = {ngram_indices[0]: 1.0}
                else: continue

                pooled_embeddings[prot_id] = pooled_emb.astype(ngram_embeddings.dtype)
            return pooled_embeddings, attention_weights_log
        else:
            raise ValueError(f"Unknown pooling strategy: '{strategy}'")

        print(f"  Pooling complete. Generated {len(pooled_embeddings)} protein embeddings.") # This line is now unreachable but kept for safety
        return pooled_embeddings, {}

    @staticmethod
    def extract_gcn_node_embeddings(model: nn.Module,
                                    full_data: Data, graph_obj: 'DirectedNgramGraph',
                                    config: 'Config', device: torch.device,
                                    create_clustered_subgraphs_func: callable) -> np.ndarray:
        """Extracts node embeddings from a GNN, handling both full-batch and clustered inference."""
        model.eval()
        model.to(device)

        if config.GCN_USE_CLUSTER_TRAINING and graph_obj.number_of_nodes > config.GCN_CLUSTER_TRAINING_THRESHOLD_NODES:
            print(f"  Extracting embeddings for {graph_obj.number_of_nodes} nodes using clustered inference...")
            partitions = create_clustered_subgraphs_func(graph_obj)
            if not partitions:
                return np.array([])

            results = []
            for node_idx_batch in tqdm(partitions, desc="  Inference on subgraphs", leave=False):
                nodes_tensor = torch.tensor(node_idx_batch, dtype=torch.long)
                subgraph_data = graph_obj.create_subgraph_data_for_model(
                    model_type=model.__class__.__name__.lower(),
                    full_features=full_data.x,
                    full_labels=full_data.y,
                    node_subset=nodes_tensor
                ).to(device)

                with torch.no_grad():
                    _, subgraph_embeddings = model(data=subgraph_data)
                results.append((subgraph_data.original_indices.cpu(), subgraph_embeddings.cpu()))

            if not results: return np.array([])
            first_emb = results[0][1]
            all_node_embeddings = torch.zeros(graph_obj.number_of_nodes, first_emb.shape[1], dtype=first_emb.dtype)

            for indices, embeddings in results:
                all_node_embeddings[indices] = embeddings
            return all_node_embeddings.numpy()
        else:
            print(f"  Extracting embeddings for {graph_obj.number_of_nodes} nodes using full-batch inference...")
            # --- FINAL FIX: This block was incomplete and only handled 'directgcn'. ---
            # It now correctly prepares the Data object for all supported model types.
            model_type = model.__class__.__name__.lower()
            data_dict = {'x': full_data.x}

            if model_type == 'directgcn':
                data_dict.update({
                    'edge_index_in': graph_obj.mathcal_A_in.indices(), 'edge_weight_in': graph_obj.mathcal_A_in.values(),
                    'edge_index_out': graph_obj.mathcal_A_out.indices(), 'edge_weight_out': graph_obj.mathcal_A_out.values(),
                    'edge_index_undirected_norm': graph_obj.A_undirected_norm_sparse.indices(),
                    'edge_weight_undirected_norm': graph_obj.A_undirected_norm_sparse.values()
                })
            elif model_type == 'rgcn':
                # RGCN needs a combined edge_index and an edge_type tensor
                edge_index_out = graph_obj.A_out_w.indices()
                edge_index_in = graph_obj.A_in_w.indices()
                data_dict['edge_index'] = torch.cat([edge_index_out, edge_index_in], dim=1)
                data_dict['edge_type'] = torch.cat([
                    torch.zeros(edge_index_out.size(1), dtype=torch.long),
                    torch.ones(edge_index_in.size(1), dtype=torch.long)
                ])
            elif model_type == 'tongdigcn':
                # TongDiGCN needs separate forward and backward edge indices
                data_dict['edge_index'] = graph_obj.A_out_w.indices()
                data_dict['edge_index_backward'] = graph_obj.A_in_w.indices()
            else:
                # Default for standard GNNs (GCN, GAT, etc.) is the undirected normalized matrix
                data_dict['edge_index'] = graph_obj.A_undirected_norm_sparse.indices()
                data_dict['edge_attr'] = graph_obj.A_undirected_norm_sparse.values()
            # --- END FIX ---

            prepared_data = Data.from_dict(data_dict).to(device)

            with torch.no_grad():
                _, embeddings = model(data=prepared_data)
            return embeddings.cpu().numpy()

    @staticmethod
    def pool_lower_level_embeddings_for_init(
            graph_obj: 'DirectedNgramGraph',
            prev_level_embeddings: np.ndarray,
            prev_level_map: Dict[str, int],
            strategy: str = 'mean'
    ) -> Optional[torch.Tensor]:
        """
        Generates initial features for an n-gram graph by pooling the embeddings
        of its constituent (n-1)-grams from the previous level. Supports 'mean'
        and 'attention' pooling strategies.
        """
        n_val = graph_obj.n_value
        if n_val <= 1:
            return None  # This function is only for n > 1

        print(f"  Pooling (n-1)-gram embeddings to initialize features for n={n_val} graph (Strategy: {strategy})...")
        num_nodes = graph_obj.number_of_nodes
        embedding_dim = prev_level_embeddings.shape[1]
        # Initialize with zeros; n-grams with no valid parents will have a zero-vector feature
        initial_features = np.zeros((num_nodes, embedding_dim), dtype=np.float32)

        for node_idx, ngram_str in tqdm(graph_obj.nodes.items(), desc=f"  Initializing n={n_val} features", leave=False):
            # An n-gram has two (n-1)-gram parents
            parent1_str = ngram_str[:-1]
            parent2_str = ngram_str[1:]

            parent1_idx = prev_level_map.get(parent1_str)
            parent2_idx = prev_level_map.get(parent2_str)

            if parent1_idx is not None and parent2_idx is not None:
                p1 = prev_level_embeddings[parent1_idx].astype(np.float32)
                p2 = prev_level_embeddings[parent2_idx].astype(np.float32)

                if strategy == 'attention':
                    # Use a simple attention mechanism based on the dot product with the mean
                    context_vec = (p1 + p2) / 2.0
                    score1 = np.dot(p1, context_vec)
                    score2 = np.dot(p2, context_vec)
                    # Softmax for weights
                    scores = np.array([score1, score2])
                    exp_scores = np.exp(scores - np.max(scores))  # Stabilized softmax
                    weights = exp_scores / np.sum(exp_scores)
                    initial_features[node_idx] = weights[0] * p1 + weights[1] * p2
                else:  # Default to mean pooling
                    initial_features[node_idx] = (p1 + p2) / 2.0

        return torch.from_numpy(initial_features)

    @staticmethod
    def generate_edge_features_batched(interaction_pairs: List[Tuple[str, str, int]],
                                       protein_embeddings: 'EmbeddingLoader', method: str,
                                       batch_size: int, embedding_dim: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generates edge features in batches for link prediction. Yields (features_batch, labels_batch)."""
        if not protein_embeddings or embedding_dim <= 0: return

        batch_features, batch_labels = [], []
        for p1_id, p2_id, label in interaction_pairs:
            if p1_id in protein_embeddings and p2_id in protein_embeddings:
                emb1, emb2 = protein_embeddings[p1_id], protein_embeddings[p2_id]
                if emb1.shape[0] == embedding_dim and emb2.shape[0] == embedding_dim:
                    if method == 'concatenate':
                        feature = np.concatenate((emb1, emb2))
                    elif method == 'average':
                        feature = (emb1.astype(np.float32) + emb2.astype(np.float32)) / 2.0
                    elif method == 'hadamard':
                        feature = emb1 * emb2
                    elif method == 'l1_distance':
                        feature = np.abs(emb1 - emb2)
                    elif method == 'l2_distance':
                        feature = (emb1 - emb2) ** 2
                    else:
                        feature = np.concatenate((emb1, emb2))

                    batch_features.append(feature.astype(np.float16))
                    batch_labels.append(label)

                    if len(batch_features) >= batch_size:
                        yield np.array(batch_features, dtype=np.float16), np.array(batch_labels, dtype=np.int32)
                        batch_features, batch_labels = [], []

        # --- FIX: Yield the final, smaller batch to ensure all data is processed. ---
        # Modern TensorFlow handles variable batch sizes gracefully, so dropping the remainder
        # is no longer necessary and leads to incomplete evaluation.
        if batch_features:
            yield np.array(batch_features, dtype=np.float16), np.array(batch_labels, dtype=np.int32)
