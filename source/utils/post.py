# ==============================================================================
# MODULE: utils/models.py
# PURPOSE: Contains tools for loading and post-processing embeddings, such as PCA,
#          normalization, pooling, and edge feature creation.
# VERSION: 7.0 (Aligned DirectGCN embedding extraction with Parallel Views architecture)
# AUTHOR: Islam Ebeid
# ==============================================================================

from pathlib import Path
from typing import Dict, Optional, List, Tuple, Set, Union, TYPE_CHECKING, Iterator, Callable

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm.auto import tqdm

if TYPE_CHECKING:
    pass


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
        # --- DEFINITIVE FIX: Use a local import to break the circular dependency with data.py ---
        from .data import DataUtils

        print(f"Applying PCA to file: '{input_h5_path.name}'. Target dimension: {target_dimension}")
        output_dir.mkdir(parents=True, exist_ok=True)
        # --- FIX: Use a more robust and specific suffix for PCA files ---
        output_h5_path = output_dir / f"{input_h5_path.stem}.pca_{target_dimension}.h5"

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
        # --- FIX: More memory-efficient implementation ---
        # Pre-allocate the array and fill it, avoiding a large intermediate list of arrays.
        valid_residues = [res for res in sequence if res in w2v_model.wv]
        if not valid_residues:
            return np.zeros((0, embedding_dim), dtype=np.float32)

        residue_vectors = np.zeros((len(valid_residues), embedding_dim), dtype=np.float32)
        for i, residue in enumerate(valid_residues):
            residue_vectors[i] = w2v_model.wv[residue]
        return residue_vectors

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
                # --- FIX: Use numerically stable softmax (subtract max score) ---
                # This prevents overflow and is consistent with other attention implementations in this file.
                exp_scores = np.exp(attention_scores - np.max(attention_scores))
                attention_weights = exp_scores / np.sum(exp_scores)
                return np.dot(attention_weights, residue_embeddings)
            return np.mean(residue_embeddings, axis=0)  # Fallback for single-residue sequences
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

            # --- FIX: Ensure every protein has an entry, even if it's a zero-vector. ---
            # This makes the output predictable and prevents KeyErrors downstream.
            pooled_embeddings = {protein_ids_ordered[i]: protein_pooled_values[i].astype(ngram_embeddings.dtype)
                                 for i in range(num_proteins)}
            return pooled_embeddings, {}

        elif strategy == 'attention':
            # --- DEFINITIVE FIX for OOM Crash: Use a memory-efficient streaming approach for attention. ---
            # Instead of building a giant index of all n-gram occurrences for all proteins in memory,
            # this approach processes one protein at a time. This keeps memory usage low and constant,
            # bounded by the requirements of the single longest protein, not the entire dataset.
            print("    Using memory-efficient streaming method for attention pooling.")

            embedding_dim = ngram_embeddings.shape[1]
            pooled_embeddings = {}
            attention_weights_log = {}

            # Create a reverse map from index to n-gram string for logging
            idx_to_ngram_map = {v: k for k, v in ngram_map.items()}

            for prot_id, seq in tqdm(protein_sequences, desc="    Calculating Attention per Protein"):
                ngram_indices_for_this_protein = []
                if len(seq) >= n_val:
                    for i in range(len(seq) - n_val + 1):
                        ngram_idx = ngram_map.get("".join(seq[i:i + n_val]))
                        if ngram_idx is not None:
                            ngram_indices_for_this_protein.append(ngram_idx)

                if not ngram_indices_for_this_protein:
                    pooled_embeddings[prot_id] = np.zeros(embedding_dim, dtype=ngram_embeddings.dtype)
                    continue

                protein_ngrams_arr = ngram_embeddings[ngram_indices_for_this_protein].astype(np.float32)

                if protein_ngrams_arr.shape[0] > 1:
                    mean_vec = np.mean(protein_ngrams_arr, axis=0, keepdims=True)
                    attention_scores = np.dot(protein_ngrams_arr, mean_vec.T).flatten()
                    exp_scores = np.exp(attention_scores - np.max(attention_scores))
                    attention_weights = exp_scores / np.sum(exp_scores)
                    pooled_emb = np.dot(attention_weights, protein_ngrams_arr)  # Log the weights by their n-gram STRING

                    attention_weights_log[prot_id] = {
                        idx_to_ngram_map.get(idx, str(idx)): float(weight)
                        for idx, weight in zip(ngram_indices_for_this_protein, attention_weights)
                    }
                elif protein_ngrams_arr.shape[0] == 1:
                    pooled_emb = protein_ngrams_arr[0]
                    # Attention for a single item is 1.0, log with its string representation
                    attention_weights_log[prot_id] = {idx_to_ngram_map.get(ngram_indices_for_this_protein[0], str(ngram_indices_for_this_protein[0])): 1.0}
                else:
                    pooled_emb = np.zeros(embedding_dim, dtype=np.float32)
                    continue

                pooled_embeddings[prot_id] = pooled_emb.astype(ngram_embeddings.dtype)
            return pooled_embeddings, attention_weights_log
        raise ValueError(f"Unknown pooling strategy: '{strategy}'")

    @staticmethod
    def extract_gcn_node_embeddings(model: nn.Module,
                                    full_data: Data, graph_obj: 'DirectedNgramGraph',
                                    config: 'Config', device: torch.device,
                                    prepare_data_func: Callable,
                                    create_clustered_subgraphs_func: callable) -> np.ndarray:
        """Extracts node embeddings from a GNN, handling both full-batch and clustered inference."""
        model.eval()
        model.to(device)

        if config.PROTGRAM_USE_CLUSTER_TRAINING and graph_obj.number_of_nodes > config.PROTGRAM_CLUSTER_TRAINING_THRESHOLD_NODES:
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
                    node_subset=nodes_tensor,
                    # --- DEFINITIVE FIX: Pass the homophily/heterophily matrices to the subgraph creator ---
                    A_homo_norm=getattr(full_data, 'A_homo_norm', None),
                    A_hetero_norm=getattr(full_data, 'A_hetero_norm', None)
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
            # --- DEFINITIVE FIX: Use the provided data preparation function ---
            # This ensures that data for inference is prepared identically to how it was for training.
            prepared_data = prepare_data_func(
                model_type=model.__class__.__name__.lower(),
                graph=graph_obj, features=full_data.x, labels=full_data.y
            ).to(device)

            with torch.no_grad():
                _, embeddings = model(data=prepared_data)
            return embeddings.cpu().numpy()

    @staticmethod
    def pool_lower_level_embeddings_for_init(
            graph_obj: 'DirectedNgramGraph',
            prev_level_embeddings: np.ndarray,
            prev_level_map: Dict[str, int],
            strategy: str = 'mean'
    ) -> Optional[Tuple[torch.Tensor, Dict[str, Dict[str, float]]]]:
        """
        Generates initial features for an n-gram graph by pooling the embeddings
        of its constituent (n-1)-grams from the previous level. Supports 'mean'
        and 'attention' pooling strategies.

        Returns: A tuple of (features_tensor, attention_weights_dict).
        """
        n_val = graph_obj.n_value
        if n_val <= 1:
            return None  # This function is only for n > 1

        print(f"  Pooling (n-1)-gram embeddings to initialize features for n={n_val} graph (Strategy: {strategy})...")
        num_nodes = graph_obj.number_of_nodes
        embedding_dim = prev_level_embeddings.shape[1]
        # Initialize with zeros; n-grams with no valid parents will have a zero-vector feature
        hierarchical_attention_log: Dict[str, Dict[str, float]] = {}
        initial_features = np.zeros((num_nodes, embedding_dim), dtype=np.float32)

        for node_idx, ngram_str in tqdm(graph_obj.idx_to_node.items(), desc=f"  Initializing n={n_val} features", leave=False):
            # An n-gram has two (n-1)-gram parents
            parent1_str = ngram_str[:-1]
            parent2_str = ngram_str[1:]

            parent1_idx = prev_level_map.get(parent1_str)
            parent2_idx = prev_level_map.get(parent2_str)

            valid_parents = []
            if parent1_idx is not None:
                valid_parents.append(prev_level_embeddings[parent1_idx].astype(np.float32))
            if parent2_idx is not None:
                valid_parents.append(prev_level_embeddings[parent2_idx].astype(np.float32))

            if len(valid_parents) == 2:
                p1, p2 = valid_parents[0], valid_parents[1]
                if strategy == 'attention':
                    # Use a simple attention mechanism based on the dot product with the mean
                    context_vec = (p1 + p2) / 2.0
                    score1 = np.dot(p1, context_vec)
                    score2 = np.dot(p2, context_vec)
                    # Softmax for weights
                    scores = np.array([score1, score2])
                    exp_scores = np.exp(scores - np.max(scores))  # Stabilized softmax
                    weights = exp_scores / np.sum(exp_scores)
                    initial_features[node_idx] = (weights[0] * p1) + (weights[1] * p2)
                    # Log the attention weights
                    hierarchical_attention_log[ngram_str] = {
                        parent1_str: float(weights[0]),
                        parent2_str: float(weights[1])
                    }
                else:  # Default to mean pooling
                    initial_features[node_idx] = (p1 + p2) / 2.0
            elif len(valid_parents) == 1:
                # --- FIX: If only one parent is valid, use its embedding directly ---
                initial_features[node_idx] = valid_parents[0]

        return torch.from_numpy(initial_features), hierarchical_attention_log

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
