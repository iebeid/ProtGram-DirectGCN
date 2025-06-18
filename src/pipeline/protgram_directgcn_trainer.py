# src/pipeline/protgram_directgcn_trainer.py
# ==============================================================================
# MODULE: pipeline/protgram_directgcn_trainer.py
# PURPOSE: Trains the ProtGramDirectGCN model, saves embeddings, and optionally
#          applies PCA for dimensionality reduction.
# VERSION: 3.7 (Corrected L2 regularization accumulation in _train_model)
# ==============================================================================

import collections
import gc
import os
import random
from functools import partial
from multiprocessing import Pool
from typing import Dict, Tuple, Optional

import h5py
import numpy as np
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

from config import Config
from src.models.protgram_directgcn import ProtGramDirectGCN
from src.utils.data_utils import DataLoader, DataUtils
from src.utils.graph_utils import DirectedNgramGraph  # Ensure this is the updated version
from src.utils.models_utils import EmbeddingProcessor

AMINO_ACID_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


class ProtGramDirectGCNTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.id_map: Dict[str, str] = {}
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)
        DataUtils.print_header("ProtGramDirectGCNEmbedder Initialized")

    def _train_model(self, model: ProtGramDirectGCN, data: Data, optimizer: torch.optim.Optimizer, epochs: int, l2_lambda: float = 0.0):
        model.train()
        model.to(self.device)
        data = data.to(self.device)  # Ensure all parts of data are on device
        criterion = torch.nn.NLLLoss()
        targets = data.y
        mask = getattr(data, 'train_mask', torch.ones(data.num_nodes, dtype=torch.bool, device=self.device))

        if mask.sum() == 0:
            print("  Warning: No valid training samples found based on the mask.")
            return

        print(f"  Starting model training for {epochs} epochs (L2 lambda: {l2_lambda})...")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            log_probs, _ = model(data=data)

            if not hasattr(log_probs, 'size') or log_probs[mask].size(0) == 0:
                if epoch == 1: print(f"    Warning: Mask resulted in 0 training samples for log_probs in epoch {epoch}.")
                continue
            if not hasattr(targets, 'size') or targets[mask].size(0) == 0:
                if epoch == 1: print(f"    Warning: Mask resulted in 0 training samples for targets in epoch {epoch}.")
                continue
            if log_probs[mask].size(0) != targets[mask].size(0):
                print(f"    Warning: Mismatch in masked log_probs ({log_probs[mask].size(0)}) and targets ({targets[mask].size(0)}) in epoch {epoch}.")
                continue

            primary_loss = criterion(log_probs[mask], targets[mask].to(log_probs.device).long())

            # --- MODIFIED L2 Regularization Calculation ---
            final_l2_reg_term = torch.tensor(0., device=self.device)
            if l2_lambda > 0:
                param_sq_norms = []
                for param in model.parameters():
                    if param.requires_grad:  # Only regularize parameters that require gradients
                        param_sq_norms.append(torch.norm(param, p=2).pow(2))

                if param_sq_norms:  # If there are any parameters to regularize
                    final_l2_reg_term = torch.stack(param_sq_norms).sum()
            # --- END MODIFIED L2 Regularization Calculation ---

            loss = primary_loss + l2_lambda * final_l2_reg_term  # Use the summed term

            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            # else:
            # if epoch == 1: print(f"    Warning: Loss does not require grad in epoch {epoch}. Skipping backward/step.")

            if epoch % (max(1, epochs // 10)) == 0 or epoch == epochs:
                if self.config.DEBUG_VERBOSE:
                    # Use final_l2_reg_term for logging the L2 component's contribution
                    print(f"    Epoch: {epoch:03d}, Total Loss: {loss.item():.4f}, Primary Loss: {primary_loss.item():.4f}, L2: {(l2_lambda * final_l2_reg_term).item():.4f}")
        # print("  Model training finished.")

    def _generate_community_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        import networkx as nx
        import community as community_louvain  # python-louvain
        graph_n_value_str = f"n={graph.n_value if hasattr(graph, 'n_value') else 'Unknown'}"
        # print(f"  Generating 'community' labels for graph {graph_n_value_str}...")
        if graph.number_of_nodes == 0: return torch.empty(0, dtype=torch.long), 1

        combined_adj_torch = graph.A_in_w + graph.A_out_w
        combined_adj_np = combined_adj_torch.cpu().numpy()
        combined_adj_sparse = csr_matrix(combined_adj_np)

        if combined_adj_sparse.nnz == 0:
            # print(f"    Warning: Graph for {graph_n_value_str} has no weighted edges for community detection. Assigning all nodes to a single community (0).")
            return torch.zeros(graph.number_of_nodes, dtype=torch.long), 1

        nx_graph = nx.from_scipy_sparse_array(combined_adj_sparse)
        if nx_graph.number_of_nodes() == 0: return torch.empty(0, dtype=torch.long), 1

        partition = community_louvain.best_partition(nx_graph, random_state=self.config.RANDOM_STATE)
        labels_list = [partition.get(i, -1) for i in range(graph.number_of_nodes)]
        unique_labels_from_partition = sorted(list(set(labels_list)))

        if not unique_labels_from_partition or (len(unique_labels_from_partition) == 1 and unique_labels_from_partition[0] == -1):
            labels = torch.zeros(graph.number_of_nodes, dtype=torch.long)
            num_classes = 1
        else:
            label_map = {lbl: i for i, lbl in enumerate(unique_labels_from_partition)}
            labels = torch.tensor([label_map[lbl] for lbl in labels_list], dtype=torch.long)
            num_classes = len(unique_labels_from_partition)
        # print(f"    Detected {num_classes} communities for {graph_n_value_str} using Louvain algorithm.")
        return labels, num_classes

    def _generate_next_node_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        num_nodes = graph.number_of_nodes
        graph_n_value_str = f"n={graph.n_value if hasattr(graph, 'n_value') else 'Unknown'}"
        # print(f"  Generating 'next_node' labels for graph {graph_n_value_str} (using A_out_w)...")
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), 1

        adj_out_weighted = graph.A_out_w  # This is a torch.Tensor
        labels_list = [-1] * num_nodes
        for i in range(num_nodes):
            successors_indices = (adj_out_weighted[i, :] > 0).nonzero(as_tuple=True)[0]
            if successors_indices.numel() > 0:
                weights = adj_out_weighted[i, successors_indices]
                max_weight = torch.max(weights)
                highest_prob_successors = successors_indices[weights == max_weight]
                labels_list[i] = random.choice(highest_prob_successors.cpu().tolist())
            else:
                labels_list[i] = i
        final_labels = torch.tensor(labels_list, dtype=torch.long)
        # print(f"  Finished 'next_node' label generation. Task output classes: {num_nodes}.")
        return final_labels, num_nodes

    def _generate_closest_amino_acid_labels(self, graph: DirectedNgramGraph, k_hops: int) -> Tuple[torch.Tensor, int]:
        num_nodes = graph.number_of_nodes
        adj_out_for_bfs = graph.A_out_w  # torch.Tensor

        if not hasattr(graph, 'node_sequences') or not graph.node_sequences:
            if hasattr(graph, 'idx_to_node') and graph.idx_to_node:
                graph.node_sequences = [graph.idx_to_node[i] for i in range(num_nodes)]
            else:
                raise AttributeError("Graph needs 'node_sequences' or 'idx_to_node' populated with actual n-gram strings.")

        node_sequences = graph.node_sequences
        graph_n_value_str = f"n={graph.n_value if hasattr(graph, 'n_value') else 'Unknown'}"
        # print(f"  Generating 'closest_amino_acid' labels for graph {graph_n_value_str} (k_hops={k_hops})...")
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), k_hops + 1
        if k_hops < 0: raise ValueError("k_hops must be non-negative.")

        labels_for_nodes = torch.full((num_nodes,), k_hops, dtype=torch.long)

        for start_node_idx in range(num_nodes):  # Removed tqdm for less verbose default
            target_aa = random.choice(AMINO_ACID_ALPHABET)
            if target_aa in node_sequences[start_node_idx]:
                labels_for_nodes[start_node_idx] = 0
                continue

            if k_hops > 0:
                q = collections.deque([(start_node_idx, 0)])
                visited = {start_node_idx}
                found_at_hop = -1
                while q:
                    curr_node, hop_level = q.popleft()
                    if hop_level >= k_hops: continue

                    neighbors_indices = (adj_out_for_bfs[curr_node, :] > 0).nonzero(as_tuple=True)[0]
                    for neighbor_node_tensor in neighbors_indices:
                        neighbor_node = neighbor_node_tensor.item()
                        if neighbor_node not in visited:
                            visited.add(neighbor_node)
                            if target_aa in node_sequences[neighbor_node]:
                                labels_for_nodes[start_node_idx] = hop_level + 1
                                found_at_hop = hop_level + 1
                                break
                            q.append((neighbor_node, hop_level + 1))
                    if found_at_hop != -1: break

        num_output_classes = k_hops + 1
        # print(f"  Finished 'closest_amino_acid' label generation. Task output classes: {num_output_classes}.")
        return labels_for_nodes, num_output_classes

    def run(self):
        DataUtils.print_header("PIPELINE STEP 2: Training ProtGramDirectGCN & Generating Embeddings")
        os.makedirs(self.config.GCN_EMBEDDINGS_DIR, exist_ok=True)

        DataUtils.print_header("Step 1: Loading Protein ID Mapping (if configured)")
        if self.config.ID_MAPPING_MODE != 'none':
            id_mapper_instance = DataLoader(config=self.config)
            self.id_map = id_mapper_instance.generate_id_maps()
            print(f"  Loaded {len(self.id_map)} ID mappings.")
        else:
            print("  ID mapping mode is 'none'. Using original FASTA IDs.")
            self.id_map = {}

        print(f"Using device: {self.device}")

        level_embeddings: Dict[int, np.ndarray] = {}
        level_ngram_to_idx: Dict[int, Dict[str, int]] = {}
        l2_lambda_val = getattr(self.config, 'GCN_L2_REG_LAMBDA', 0.0)

        for n_val in range(1, self.config.GCN_NGRAM_MAX_N + 1):
            DataUtils.print_header(f"Processing N-gram Level: n = {n_val}")
            graph_path = os.path.join(self.config.GRAPH_OBJECTS_DIR, f"ngram_graph_n{n_val}.pkl")

            graph_obj: Optional[DirectedNgramGraph] = None
            try:
                print(f"  Loading graph object from: {graph_path}")
                loaded_data = DataUtils.load_object(graph_path)
                if isinstance(loaded_data, DirectedNgramGraph):
                    graph_obj = loaded_data
                    graph_obj.epsilon_propagation = self.gcn_propagation_epsilon
                    if graph_obj.number_of_nodes > 0:
                        graph_obj._create_raw_weighted_adj_matrices_torch()
                        graph_obj._create_propagation_matrices_for_gcn()
                        graph_obj._create_symmetrized_magnitudes_fai_fao()
                elif isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                    nodes_data, edges_data = loaded_data
                    graph_obj = DirectedNgramGraph(nodes_data, edges_data, epsilon_propagation=self.gcn_propagation_epsilon)
                elif hasattr(loaded_data, 'nodes_map') and hasattr(loaded_data, 'original_edges'):
                    graph_obj = DirectedNgramGraph(loaded_data.nodes_map, loaded_data.original_edges, epsilon_propagation=self.gcn_propagation_epsilon)
                else:
                    raise TypeError("Loaded graph object is not in an expected format.")

                if graph_obj: graph_obj.n_value = n_val

            except FileNotFoundError:
                print(f"  ERROR: Graph object not found at {graph_path}. Skipping n={n_val}.")
                continue
            except Exception as e:
                print(f"  ERROR: Could not load or process graph for n={n_val}: {e}")
                import traceback
                traceback.print_exc()
                continue

            if not graph_obj or not hasattr(graph_obj, 'number_of_nodes'):
                print(f"  ERROR: Failed to obtain a valid graph object for n={n_val}. Skipping.")
                continue

            level_ngram_to_idx[n_val] = graph_obj.node_to_idx
            if graph_obj.number_of_nodes == 0:
                print(f"  Skipping n={n_val} (0 nodes).")
                level_embeddings[n_val] = np.array([])
                continue

            print(f"  Graph for n={n_val} loaded. Nodes: {graph_obj.number_of_nodes}")
            current_task_type = self.config.GCN_TASK_TYPES_PER_LEVEL.get(n_val, self.config.GCN_DEFAULT_TASK_TYPE)
            print(f"  Selected training task for n={n_val}: '{current_task_type}'")

            num_initial_features: int
            if n_val == 1:
                num_initial_features = self.config.GCN_1GRAM_INIT_DIM
                if num_initial_features <= 0:
                    num_initial_features = 1 if self.config.GCN_MAX_PE_LEN <= 0 else self.config.GCN_1GRAM_INIT_DIM
                    if num_initial_features <= 0: num_initial_features = 1
                    print(f"  Adjusted GCN_1GRAM_INIT_DIM for n=1 to {num_initial_features} based on PE config.")
                x = torch.randn(graph_obj.number_of_nodes, num_initial_features, device=self.device)
            else:  # n_val > 1: MODIFIED LOGIC FOR FEATURE INITIALIZATION
                if (n_val - 1) not in level_embeddings or level_embeddings[n_val - 1].size == 0:
                    print(f"  ERROR: Prev level n={n_val - 1} embeddings missing/empty for n={n_val}. Skipping.")
                    continue

                prev_level_embeds_np = level_embeddings[n_val - 1]  # These are numpy arrays
                prev_level_ngram_to_idx_map = level_ngram_to_idx[n_val - 1]

                num_initial_features = prev_level_embeds_np.shape[1]
                x = torch.zeros(graph_obj.number_of_nodes, num_initial_features, dtype=torch.float, device=self.device)

                print(f"  Initializing features for n={n_val} by pooling (n-1)-gram constituent embeddings...")

                for current_ngram_str, current_node_idx in tqdm(graph_obj.node_to_idx.items(), desc=f"  Initializing n={n_val} features", leave=False, disable=not self.config.DEBUG_VERBOSE):
                    if len(current_ngram_str) != n_val:  # Should not happen
                        if self.config.DEBUG_VERBOSE:
                            print(f"    Warning: Skipping n-gram '{current_ngram_str}' due to unexpected length for n={n_val}.")
                        continue

                    # Get the (n_val-1)-gram prefix and suffix of the current_ngram_str
                    prefix_constituent_ngram = current_ngram_str[:-1]
                    suffix_constituent_ngram = current_ngram_str[1:]

                    constituent_embeddings_to_pool = []

                    # Get embedding for prefix
                    prefix_idx = prev_level_ngram_to_idx_map.get(prefix_constituent_ngram)
                    if prefix_idx is not None and prefix_idx < len(prev_level_embeds_np):
                        constituent_embeddings_to_pool.append(prev_level_embeds_np[prefix_idx])
                    elif self.config.DEBUG_VERBOSE:
                        print(f"    Warning: Prefix '{prefix_constituent_ngram}' not found in prev (n={n_val - 1}) level map for current_ngram '{current_ngram_str}'.")

                    # Get embedding for suffix
                    suffix_idx = prev_level_ngram_to_idx_map.get(suffix_constituent_ngram)
                    if suffix_idx is not None and suffix_idx < len(prev_level_embeds_np):
                        constituent_embeddings_to_pool.append(prev_level_embeds_np[suffix_idx])
                    elif self.config.DEBUG_VERBOSE:
                        print(f"    Warning: Suffix '{suffix_constituent_ngram}' not found in prev (n={n_val - 1}) level map for current_ngram '{current_ngram_str}'.")

                    if constituent_embeddings_to_pool:
                        # Mean pool the collected embeddings
                        pooled_embedding_np = np.mean(np.array(constituent_embeddings_to_pool, dtype=np.float32), axis=0)
                        x[current_node_idx] = torch.from_numpy(pooled_embedding_np).to(self.device)
                    elif self.config.DEBUG_VERBOSE:
                        print(f"    Warning: No constituent (n-1)-gram embeddings found for n-gram '{current_ngram_str}' (idx {current_node_idx}). Initialized to zeros.")
            # END MODIFIED LOGIC

            print(f"  Initial node feature dimension for n={n_val}: {num_initial_features}")

            labels, num_classes = None, 0
            if current_task_type == "community":
                labels, num_classes = self._generate_community_labels(graph_obj)
            elif current_task_type == "next_node":
                labels, num_classes = self._generate_next_node_labels(graph_obj)
            elif current_task_type == "closest_aa":
                k_hops_for_task = getattr(self.config, 'GCN_CLOSEST_AA_K_HOPS', 3)
                labels, num_classes = self._generate_closest_amino_acid_labels(graph_obj, k_hops_for_task)
            else:
                raise ValueError(f"Unsupported GCN_TASK_TYPE '{current_task_type}' for n={n_val}.")

            if graph_obj.number_of_nodes > 0 and (labels is None or labels.numel() == 0):
                print(f"  Warning: No labels generated for n={n_val}, task '{current_task_type}'. Using zeros.")
                if num_classes == 0: num_classes = 1
                labels = torch.zeros(graph_obj.number_of_nodes, dtype=torch.long)

            labels = labels.to(self.device)

            edge_index_in, edge_weight_in = dense_to_sparse(torch.from_numpy(graph_obj.mathcal_A_in).float())
            edge_index_out, edge_weight_out = dense_to_sparse(torch.from_numpy(graph_obj.mathcal_A_out).float())

            fai_edge_index, fai_edge_weight = graph_obj.get_fai_sparse()
            fao_edge_index, fao_edge_weight = graph_obj.get_fao_sparse()

            data = Data(x=x.to(self.device),
                        y=labels.to(self.device),
                        edge_index_in=edge_index_in.to(self.device),
                        edge_weight_in=edge_weight_in.to(self.device),
                        edge_index_out=edge_index_out.to(self.device),
                        edge_weight_out=edge_weight_out.to(self.device),
                        fai_edge_index=fai_edge_index.to(self.device),
                        fai_edge_weight=fai_edge_weight.to(self.device),
                        fao_edge_index=fao_edge_index.to(self.device),
                        fao_edge_weight=fao_edge_weight.to(self.device))
            data.num_nodes = graph_obj.number_of_nodes

            full_layer_dims = [num_initial_features] + self.config.GCN_HIDDEN_LAYER_DIMS
            model = ProtGramDirectGCN(
                layer_dims=full_layer_dims,
                num_graph_nodes=graph_obj.number_of_nodes,
                task_num_output_classes=num_classes,
                n_gram_len=n_val,
                one_gram_dim=(self.config.GCN_1GRAM_INIT_DIM if n_val == 1 and self.config.GCN_1GRAM_INIT_DIM > 0 and self.config.GCN_MAX_PE_LEN > 0 else 0),
                max_pe_len=self.config.GCN_MAX_PE_LEN,
                dropout=self.config.GCN_DROPOUT_RATE,
                use_vector_coeffs=getattr(self.config, 'GCN_USE_VECTOR_COEFFS', True)
            )

            current_optimizer_weight_decay = self.config.GCN_WEIGHT_DECAY
            if l2_lambda_val > 0:
                # print(f"  Explicit L2 (lambda={l2_lambda_val}) added to loss; optimizer weight_decay {self.config.GCN_WEIGHT_DECAY} -> 0.0.")
                current_optimizer_weight_decay = 0.0
            optimizer = optim.Adam(model.parameters(), lr=self.config.GCN_LR, weight_decay=current_optimizer_weight_decay)

            self._train_model(model, data, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, l2_lambda_val)
            current_level_embeddings = EmbeddingProcessor.extract_gcn_node_embeddings(model, data, self.device)
            level_embeddings[n_val] = current_level_embeddings

            del model, data, graph_obj, optimizer, x, labels
            del edge_index_in, edge_weight_in, edge_index_out, edge_weight_out
            del fai_edge_index, fai_edge_weight, fao_edge_index, fao_edge_weight
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

        DataUtils.print_header("Step 3: Pooling N-gram Embeddings to Protein Level")
        final_n_val = self.config.GCN_NGRAM_MAX_N
        if final_n_val not in level_embeddings or level_embeddings[final_n_val].size == 0:
            print(f"ERROR: Final n={final_n_val} embeddings missing/empty. Cannot pool.")
            return

        final_ngram_embeds = level_embeddings[final_n_val]
        final_ngram_map = level_ngram_to_idx[final_n_val]

        protein_sequences_path = str(self.config.GCN_INPUT_FASTA_PATH)
        protein_sequences = list(DataLoader.parse_sequences(protein_sequences_path))

        pool_func = partial(EmbeddingProcessor.pool_ngram_embeddings_for_protein, n_val=final_n_val, ngram_map=final_ngram_map, ngram_embeddings=final_ngram_embeds)

        pooled_embeddings = {}
        num_pooling_workers = self.config.POOLING_WORKERS if self.config.POOLING_WORKERS is not None else os.cpu_count()
        if num_pooling_workers is None or num_pooling_workers < 1: num_pooling_workers = 1
        # print(f"  Using {num_pooling_workers} workers for pooling.")

        with Pool(processes=num_pooling_workers) as pool:
            for original_id, vec in tqdm(pool.imap_unordered(pool_func, protein_sequences), total=len(protein_sequences), desc="  Pooling Protein Embeddings", disable=not self.config.DEBUG_VERBOSE):
                if vec is not None:
                    final_key = self.id_map.get(original_id, original_id)
                    pooled_embeddings[final_key] = vec
        if not pooled_embeddings: print("  Warning: No protein embeddings after pooling.")

        DataUtils.print_header("Step 4: Saving Generated Embeddings")
        output_h5_path = os.path.join(self.config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings.h5")
        with h5py.File(output_h5_path, 'w') as hf:
            for key, vector in tqdm(pooled_embeddings.items(), desc="  Writing H5 File", disable=not self.config.DEBUG_VERBOSE):
                if vector is not None and vector.size > 0:
                    hf.create_dataset(key, data=vector)
        print(f"\nSUCCESS: Primary embeddings saved to: {output_h5_path}")

        if self.config.APPLY_PCA_TO_GCN and pooled_embeddings:
            DataUtils.print_header("Step 5: Applying PCA for Dimensionality Reduction")
            valid_pooled_embeddings = {k: v for k, v in pooled_embeddings.items() if v is not None and v.size > 0}
            if not valid_pooled_embeddings:
                print("  Warning: No valid embeddings to apply PCA.")
            else:
                pca_embeds = EmbeddingProcessor.apply_pca(valid_pooled_embeddings, self.config.PCA_TARGET_DIMENSION, self.config.RANDOM_STATE)
                if pca_embeds:
                    first_valid_pca_emb = next((v for v in pca_embeds.values() if v is not None and v.size > 0), None)
                    if first_valid_pca_emb is not None:
                        pca_dim = first_valid_pca_emb.shape[0]
                        pca_h5_path = os.path.join(self.config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings_pca{pca_dim}.h5")
                        with h5py.File(pca_h5_path, 'w') as hf:
                            for key, vector in tqdm(pca_embeds.items(), desc="  Writing PCA H5 File", disable=not self.config.DEBUG_VERBOSE):
                                if vector is not None and vector.size > 0: hf.create_dataset(key, data=vector)
                        print(f"  SUCCESS: PCA-reduced embeddings saved to: {pca_h5_path}")
                    else:
                        print("  PCA Warning: No valid PCA embeddings to determine dimension for saving.")
                elif pooled_embeddings:  # Check if original embeddings existed
                    print("  Warning: PCA was requested but resulted in no embeddings (apply_pca returned None).")
        DataUtils.print_header("ProtGramDirectGCN Embedding PIPELINE STEP FINISHED")
