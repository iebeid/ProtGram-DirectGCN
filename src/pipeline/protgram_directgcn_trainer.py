# src/pipeline/protgram_directgcn_trainer.py
# ==============================================================================
# MODULE: pipeline/protgram_directgcn_trainer.py
# PURPOSE: Trains the ProtGramDirectGCN model, saves embeddings, and optionally
#          applies PCA for dimensionality reduction.
# VERSION: 4.0 (Handles sparse mathcal_A from graph_utils, removed fai/fao)
# AUTHOR: Islam Ebeid
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
# from scipy.sparse import csr_matrix # No longer needed for community label generation with sparse mathcal_A
from torch_geometric.data import Data
# from torch_geometric.utils import dense_to_sparse # No longer needed for mathcal_A
from tqdm import tqdm

from config import Config
from src.models.protgram_directgcn import ProtGramDirectGCN
from src.utils.data_utils import DataLoader, DataUtils
from src.utils.graph_utils import DirectedNgramGraph
from src.utils.models_utils import EmbeddingProcessor

AMINO_ACID_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


class ProtGramDirectGCNTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.id_map: Dict[str, str] = {}
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)
        DataUtils.print_header("ProtGramDirectGCNEmbedder Initialized")

    def _train_model(self, model: ProtGramDirectGCN, data: Data, optimizer: torch.optim.Optimizer, epochs: int, l2_lambda: float = 0.0, current_n_val_for_diag: Optional[int] = None):  # Added current_n_val_for_diag
        model.train()
        model.to(self.device)
        data = data.to(self.device)
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

            final_l2_reg_term = torch.tensor(0., device=self.device)
            if l2_lambda > 0:
                param_sq_norms = []
                for param in model.parameters():
                    if param.requires_grad:
                        param_sq_norms.append(torch.norm(param, p=2).pow(2))
                if param_sq_norms:
                    final_l2_reg_term = torch.stack(param_sq_norms).sum()

            loss = primary_loss + l2_lambda * final_l2_reg_term

            if current_n_val_for_diag == 5 and epoch == 1 and self.config.DEBUG_VERBOSE:
                print(f"    DIAGNOSTIC (n=5, epoch=1):")
                print(f"      Primary Loss: {primary_loss.item():.4f}")
                print(f"      log_probs[mask] shape: {log_probs[mask].shape}")
                print(f"      targets[mask] shape: {targets[mask].shape}")
                if log_probs[mask].numel() > 0:
                    print(f"      Sample log_probs[mask][0]: {log_probs[mask][0]}")
                    print(f"      Sample targets[mask][0]: {targets[mask][0]}")

            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            if epoch % (max(1, epochs // 10)) == 0 or epoch == epochs:
                if self.config.DEBUG_VERBOSE:
                    print(f"    Epoch: {epoch:03d}, Total Loss: {loss.item():.4f}, Primary Loss: {primary_loss.item():.4f}, L2: {(l2_lambda * final_l2_reg_term).item():.4f}")

    def _generate_community_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        import networkx as nx
        import community as community_louvain
        # graph_n_value_str = f"n={graph.n_value if hasattr(graph, 'n_value') else 'Unknown'}" # Not used
        if graph.number_of_nodes == 0: return torch.empty(0, dtype=torch.long), 1

        # Combine A_in_w and A_out_w (which are sparse) for community detection
        # Ensure they are on CPU for NetworkX conversion
        A_in_w_cpu_sparse = graph.A_in_w.cpu()
        A_out_w_cpu_sparse = graph.A_out_w.cpu()

        # Add sparse matrices
        combined_adj_sparse_torch = (A_in_w_cpu_sparse + A_out_w_cpu_sparse).coalesce()

        if combined_adj_sparse_torch._nnz() == 0:
            return torch.zeros(graph.number_of_nodes, dtype=torch.long), 1

        # Convert sparse PyTorch tensor to NetworkX graph
        # Need indices and values for from_scipy_sparse_array or add_weighted_edges_from
        # A more direct way if PyG is available:
        # from torch_geometric.utils import to_networkx
        # data_for_nx = Data(edge_index=combined_adj_sparse_torch.indices(),
        #                    edge_attr=combined_adj_sparse_torch.values(),
        #                    num_nodes=graph.number_of_nodes)
        # nx_graph = to_networkx(data_for_nx, edge_attrs=['edge_attr'], to_undirected=True)
        # For now, let's build it manually to avoid new dependencies if not already used

        nx_graph = nx.Graph()  # Louvain works on undirected graphs
        nx_graph.add_nodes_from(range(graph.number_of_nodes))
        edge_indices = combined_adj_sparse_torch.indices()
        edge_values = combined_adj_sparse_torch.values()

        for i in range(edge_indices.shape[1]):
            u, v = edge_indices[0, i].item(), edge_indices[1, i].item()
            weight = edge_values[i].item()
            if nx_graph.has_edge(u, v):
                nx_graph[u][v]['weight'] += weight
            else:
                nx_graph.add_edge(u, v, weight=weight)

        if nx_graph.number_of_nodes() == 0: return torch.empty(0, dtype=torch.long), 1
        if nx_graph.number_of_edges() == 0:  # If no edges, all nodes are their own community
            labels = torch.arange(graph.number_of_nodes, dtype=torch.long)
            num_classes = graph.number_of_nodes if graph.number_of_nodes > 0 else 1
            return labels, num_classes

        partition = community_louvain.best_partition(nx_graph, random_state=self.config.RANDOM_STATE, weight='weight')
        labels_list = [partition.get(i, -1) for i in range(graph.number_of_nodes)]
        unique_labels_from_partition = sorted(list(set(labels_list)))

        if not unique_labels_from_partition or (len(unique_labels_from_partition) == 1 and unique_labels_from_partition[0] == -1):
            labels = torch.zeros(graph.number_of_nodes, dtype=torch.long)
            num_classes = 1
        else:
            label_map = {lbl: i for i, lbl in enumerate(unique_labels_from_partition)}
            labels = torch.tensor([label_map[lbl] for lbl in labels_list], dtype=torch.long)
            num_classes = len(unique_labels_from_partition)

        if hasattr(graph, 'n_value') and graph.n_value == 5 and self.config.DEBUG_VERBOSE:
            print(f"    DIAGNOSTIC (n=5 Community Labels):")
            print(f"      Number of communities detected (num_classes): {num_classes}")
            print(f"      Unique labels in generated 'labels' tensor: {labels.unique().tolist() if labels.numel() > 0 else '[]'}")
        return labels, num_classes

    def _generate_next_node_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), 1

        adj_out_weighted_sparse = graph.A_out_w  # This is sparse
        labels_list = [-1] * num_nodes

        for i in range(num_nodes):
            # Get outgoing edges for node i
            # A_out_w is (num_nodes, num_nodes). Row i corresponds to outgoing edges from node i.
            # For a sparse COO tensor, we need to find entries where indices()[0] == i
            row_mask = (adj_out_weighted_sparse.indices()[0] == i)
            if not torch.any(row_mask):  # No outgoing edges
                labels_list[i] = i  # Self-loop as target
                continue

            successors_indices_for_row_i = adj_out_weighted_sparse.indices()[1][row_mask]
            weights_for_row_i = adj_out_weighted_sparse.values()[row_mask]

            if successors_indices_for_row_i.numel() > 0:
                max_weight = torch.max(weights_for_row_i)
                highest_prob_successors = successors_indices_for_row_i[weights_for_row_i == max_weight]
                labels_list[i] = random.choice(highest_prob_successors.cpu().tolist())
            else:  # Should have been caught by torch.any(row_mask)
                labels_list[i] = i

        final_labels = torch.tensor(labels_list, dtype=torch.long)
        return final_labels, num_nodes  # num_classes is num_nodes

    def _generate_closest_amino_acid_labels(self, graph: DirectedNgramGraph, k_hops: int) -> Tuple[torch.Tensor, int]:
        num_nodes = graph.number_of_nodes
        adj_out_sparse = graph.A_out_w  # This is sparse

        if not hasattr(graph, 'node_sequences') or not graph.node_sequences:
            if hasattr(graph, 'idx_to_node') and graph.idx_to_node:
                graph.node_sequences = [graph.idx_to_node[i] for i in range(num_nodes)]
            else:
                raise AttributeError("Graph needs 'node_sequences' or 'idx_to_node' populated with actual n-gram strings.")

        node_sequences = graph.node_sequences
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), k_hops + 1
        if k_hops < 0: raise ValueError("k_hops must be non-negative.")

        labels_for_nodes = torch.full((num_nodes,), k_hops, dtype=torch.long)  # Default to max_hops (target not found within k)

        for start_node_idx in range(num_nodes):
            target_aa = random.choice(AMINO_ACID_ALPHABET)

            current_node_sequence = node_sequences[start_node_idx]
            if not isinstance(current_node_sequence, str): current_node_sequence = str(current_node_sequence)

            if target_aa in current_node_sequence:
                labels_for_nodes[start_node_idx] = 0
                continue

            if k_hops > 0:
                q = collections.deque([(start_node_idx, 0)])
                visited = {start_node_idx}
                found_at_hop = -1  # Flag to break outer loop once found

                while q:
                    curr_node_q_idx, hop_level = q.popleft()
                    if hop_level >= k_hops:  # Already at max depth for this path
                        continue

                    # Get neighbors for curr_node_q_idx from sparse adj_out_sparse
                    row_mask_bfs = (adj_out_sparse.indices()[0] == curr_node_q_idx)
                    if not torch.any(row_mask_bfs):
                        continue

                    neighbors_indices_for_curr = adj_out_sparse.indices()[1][row_mask_bfs]

                    for neighbor_node_tensor in neighbors_indices_for_curr:
                        neighbor_node_idx = neighbor_node_tensor.item()
                        if neighbor_node_idx not in visited:
                            visited.add(neighbor_node_idx)

                            neighbor_node_sequence = node_sequences[neighbor_node_idx]
                            if not isinstance(neighbor_node_sequence, str): neighbor_node_sequence = str(neighbor_node_sequence)

                            if target_aa in neighbor_node_sequence:
                                labels_for_nodes[start_node_idx] = hop_level + 1
                                found_at_hop = hop_level + 1
                                break  # Found for this start_node_idx, break from neighbor loop

                            if hop_level + 1 < k_hops:  # Only add to queue if not exceeding max hops
                                q.append((neighbor_node_idx, hop_level + 1))

                    if found_at_hop != -1:
                        break  # Break from BFS queue loop for this start_node_idx

        num_output_classes = k_hops + 1  # Classes are 0, 1, ..., k_hops
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
                    # Re-initialize propagation matrices as they are not pickled with sparse tensors well by default
                    # and to ensure correct device placement if loaded object was on CPU.
                    if graph_obj.number_of_nodes > 0:
                        # Ensure A_out_w and A_in_w are on the correct device before recomputing mathcal_A
                        # This assumes _create_raw_weighted_adj_matrices_torch sets them to CPU or a default device
                        # If they are already torch tensors, move them.
                        if hasattr(graph_obj, 'A_out_w') and isinstance(graph_obj.A_out_w, torch.Tensor):
                            graph_obj.A_out_w = graph_obj.A_out_w.to(self.device)
                        if hasattr(graph_obj, 'A_in_w') and isinstance(graph_obj.A_in_w, torch.Tensor):
                            graph_obj.A_in_w = graph_obj.A_in_w.to(self.device)

                        # If A_out_w/A_in_w were not properly initialized or are not sparse tensors,
                        # re-run _create_raw_weighted_adj_matrices_torch.
                        # This is a safeguard. Ideally, the pickled object should be consistent.
                        if not (hasattr(graph_obj, 'A_out_w') and graph_obj.A_out_w.is_sparse) or \
                                not (hasattr(graph_obj, 'A_in_w') and graph_obj.A_in_w.is_sparse):
                            print("    Re-creating raw weighted adjacency matrices for loaded graph object...")
                            graph_obj._create_raw_weighted_adj_matrices_torch()  # This will use self.device internally if A_out_w is created new

                        # Always re-create propagation matrices after loading
                        print("    Re-creating propagation matrices for loaded graph object...")
                        graph_obj._create_propagation_matrices_for_gcn()
                else:
                    raise TypeError(f"Loaded graph object is not of type DirectedNgramGraph, but {type(loaded_data)}.")

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
            else:
                if (n_val - 1) not in level_embeddings or level_embeddings[n_val - 1].size == 0:
                    print(f"  ERROR: Prev level n={n_val - 1} embeddings missing/empty for n={n_val}. Skipping.")
                    continue
                prev_level_embeds_np = level_embeddings[n_val - 1]
                prev_level_ngram_to_idx_map = level_ngram_to_idx[n_val - 1]
                num_initial_features = prev_level_embeds_np.shape[1]
                x = torch.zeros(graph_obj.number_of_nodes, num_initial_features, dtype=torch.float, device=self.device)
                print(f"  Initializing features for n={n_val} by pooling (n-1)-gram constituent embeddings...")
                for current_ngram_str, current_node_idx in tqdm(graph_obj.node_to_idx.items(), desc=f"  Initializing n={n_val} features", leave=False, disable=not self.config.DEBUG_VERBOSE):
                    if len(current_ngram_str) != n_val:
                        if self.config.DEBUG_VERBOSE: print(f"    Warning: Skipping n-gram '{current_ngram_str}' due to unexpected length for n={n_val}.")
                        continue
                    prefix_constituent_ngram = current_ngram_str[:-1]
                    suffix_constituent_ngram = current_ngram_str[1:]
                    constituent_embeddings_to_pool = []
                    prefix_idx = prev_level_ngram_to_idx_map.get(prefix_constituent_ngram)
                    if prefix_idx is not None and prefix_idx < len(prev_level_embeds_np):
                        constituent_embeddings_to_pool.append(prev_level_embeds_np[prefix_idx])
                    # elif self.config.DEBUG_VERBOSE:
                    # print(f"    Warning: Prefix '{prefix_constituent_ngram}' not found in prev (n={n_val - 1}) level map for current_ngram '{current_ngram_str}'.")
                    suffix_idx = prev_level_ngram_to_idx_map.get(suffix_constituent_ngram)
                    if suffix_idx is not None and suffix_idx < len(prev_level_embeds_np):
                        constituent_embeddings_to_pool.append(prev_level_embeds_np[suffix_idx])
                    # elif self.config.DEBUG_VERBOSE:
                    # print(f"    Warning: Suffix '{suffix_constituent_ngram}' not found in prev (n={n_val - 1}) level map for current_ngram '{current_ngram_str}'.")
                    if constituent_embeddings_to_pool:
                        pooled_embedding_np = np.mean(np.array(constituent_embeddings_to_pool, dtype=np.float32), axis=0)
                        x[current_node_idx] = torch.from_numpy(pooled_embedding_np).to(self.device)
                    # elif self.config.DEBUG_VERBOSE:
                    # print(f"    Warning: No constituent (n-1)-gram embeddings found for n-gram '{current_ngram_str}' (idx {current_node_idx}). Initialized to zeros.")

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

            if num_classes == 0 and graph_obj.number_of_nodes > 0:  # Safety net
                print(f"  Warning: num_classes is 0 for n={n_val}, task '{current_task_type}' despite having nodes. Setting to 1.")
                num_classes = 1
                if labels is not None and labels.max() >= num_classes:  # Adjust labels if they exceed new num_classes
                    labels = torch.zeros_like(labels)

            if n_val == 5 and current_task_type == "community" and self.config.DEBUG_VERBOSE:
                print(f"    DIAGNOSTIC (n=5, before model init):")
                print(f"      Task type: {current_task_type}")
                print(f"      Number of classes for model: {num_classes}")
                if labels is not None:
                    print(f"      Shape of labels tensor: {labels.shape}")
                    print(f"      Unique labels passed to Data object: {labels.unique().tolist() if labels.numel() > 0 else '[]'}")

            # mathcal_A_out and mathcal_A_in are now sparse torch.Tensor
            # Their indices and values are directly used for PyG Data object
            edge_index_in = graph_obj.mathcal_A_in.indices().to(self.device)
            edge_weight_in = graph_obj.mathcal_A_in.values().to(self.device)
            edge_index_out = graph_obj.mathcal_A_out.indices().to(self.device)
            edge_weight_out = graph_obj.mathcal_A_out.values().to(self.device)

            # fai and fao are removed
            data = Data(x=x.to(self.device),
                        y=labels.to(self.device),
                        edge_index_in=edge_index_in,
                        edge_weight_in=edge_weight_in,
                        edge_index_out=edge_index_out,
                        edge_weight_out=edge_weight_out)
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
                current_optimizer_weight_decay = 0.0  # L2 reg handled in loss
            optimizer = optim.Adam(model.parameters(), lr=self.config.GCN_LR, weight_decay=current_optimizer_weight_decay)

            self._train_model(model, data, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, l2_lambda_val, current_n_val_for_diag=n_val)
            current_level_embeddings = EmbeddingProcessor.extract_gcn_node_embeddings(model, data, self.device)
            level_embeddings[n_val] = current_level_embeddings

            del model, data, graph_obj, optimizer, x, labels
            del edge_index_in, edge_weight_in, edge_index_out, edge_weight_out
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

        # Limit pooling workers if num_sequences is small to avoid overhead
        effective_pooling_workers = min(num_pooling_workers, len(protein_sequences)) if len(protein_sequences) > 0 else 1

        if effective_pooling_workers > 1 and len(protein_sequences) > effective_pooling_workers:  # Only use Pool if beneficial
            with Pool(processes=effective_pooling_workers) as pool:
                for original_id, vec in tqdm(pool.imap_unordered(pool_func, protein_sequences), total=len(protein_sequences), desc="  Pooling Protein Embeddings (Parallel)", disable=not self.config.DEBUG_VERBOSE):
                    if vec is not None:
                        final_key = self.id_map.get(original_id, original_id)
                        pooled_embeddings[final_key] = vec
        else:  # Sequential pooling for small number of sequences or if workers=1
            print(f"  Using sequential pooling (workers: {effective_pooling_workers}, sequences: {len(protein_sequences)}).")
            for seq_data in tqdm(protein_sequences, desc="  Pooling Protein Embeddings (Sequential)", disable=not self.config.DEBUG_VERBOSE):
                original_id, vec = pool_func(seq_data)
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
                elif pooled_embeddings:  # Check original pooled_embeddings as pca_embeds might be None if PCA failed
                    print("  Warning: PCA was requested but resulted in no embeddings (apply_pca returned None or empty dict).")
        DataUtils.print_header("ProtGramDirectGCN Embedding PIPELINE STEP FINISHED")
