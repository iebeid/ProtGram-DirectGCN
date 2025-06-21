# src/pipeline/protgram_directgcn_trainer.py
# ==============================================================================
# MODULE: pipeline/protgram_directgcn_trainer.py
# PURPOSE: Trains the ProtGramDirectGCN model, saves embeddings, and optionally
#          applies PCA for dimensionality reduction.
# VERSION: 4.15 (Integrates new undirected matrix from graph_obj into Data object)
# AUTHOR: Islam Ebeid
# ==============================================================================

import collections
import gc
import os
import random
import math
from typing import Dict, Tuple, Optional, List
from functools import partial

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx
from tqdm import tqdm

from config import Config
from src.models.protgram_directgcn import ProtGramDirectGCN
from src.utils.data_utils import DataLoader, DataUtils, GroundTruthLoader
from src.utils.graph_utils import DirectedNgramGraph
from src.utils.models_utils import EmbeddingProcessor, EmbeddingLoader

# --- Optional Imports for Sanity Check PPI Task ---
try:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from src.models.mlp import MLP

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
# --- End Optional Imports ---

AMINO_ACID_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")


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


class ProtGramDirectGCNTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.id_map: Dict[str, str] = {}
        self.gcn_propagation_epsilon = getattr(config, 'GCN_PROPAGATION_EPSILON', 1e-9)
        DataUtils.print_header("ProtGramDirectGCNEmbedder Initialized")

    def _train_model_full_batch(self, model: ProtGramDirectGCN, data: Data, optimizer: torch.optim.Optimizer, epochs: int,
                                task_type: str, l2_lambda: float = 0.0):
        model.train()
        model.to(self.device)
        data = data.to(self.device)
        scheduler = None
        if self.config.GCN_USE_LR_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.GCN_LR_SCHEDULER_PATIENCE, factor=self.config.GCN_LR_SCHEDULER_FACTOR, verbose=self.config.DEBUG_VERBOSE)
        early_stopper = None
        if self.config.GCN_USE_EARLY_STOPPING:
            early_stopper = EarlyStopper(patience=self.config.GCN_EARLY_STOPPING_PATIENCE, min_delta=self.config.GCN_EARLY_STOPPING_MIN_DELTA)
        scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        criterion = F.nll_loss
        print(f"  Starting full-batch training for up to {epochs} epochs (Task: {task_type}, L2 lambda: {l2_lambda})...")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                task_output, _ = model(data=data)
                primary_loss = criterion(task_output, data.y)
                l2_reg = sum(p.norm(2).pow(2) for p in model.parameters() if p.requires_grad)
                loss = primary_loss + l2_lambda * l2_reg
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step(loss)
            if epoch == epochs: # Only print for the last epoch
                if self.config.DEBUG_VERBOSE:
                    print(f"    Epoch: {epoch:03d}, Total Loss: {loss.item():.4f}, Primary Loss: {primary_loss.item():.4f}, L2: {(l2_lambda * l2_reg).item():.4f}")
            if early_stopper and early_stopper.early_stop(loss.item()):
                print(f"  Early stopping triggered at epoch {epoch}. Best loss: {early_stopper.best_loss:.4f}")
                break

    def _train_model_clustered(self, model: ProtGramDirectGCN, subgraphs: List[Data], optimizer: torch.optim.Optimizer,
                               epochs: int, task_type: str, l2_lambda: float = 0.0,
                               total_nodes_in_level_graph: int = 1):
        model.train()
        model.to(self.device)
        scheduler = None
        if self.config.GCN_USE_LR_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.GCN_LR_SCHEDULER_PATIENCE, factor=self.config.GCN_LR_SCHEDULER_FACTOR, verbose=self.config.DEBUG_VERBOSE)
        early_stopper = None
        if self.config.GCN_USE_EARLY_STOPPING:
            early_stopper = EarlyStopper(patience=self.config.GCN_EARLY_STOPPING_PATIENCE, min_delta=self.config.GCN_EARLY_STOPPING_MIN_DELTA)
        scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        criterion = F.nll_loss
        print(f"  Starting Cluster-GCN style training for up to {epochs} epochs on {len(subgraphs)} subgraphs (Task: {task_type})...")
        for epoch in range(1, epochs + 1):
            random.shuffle(subgraphs)
            epoch_loss = 0.0
            for batch_data in tqdm(subgraphs, desc=f"  Epoch {epoch}", leave=False, disable=not self.config.DEBUG_VERBOSE):
                batch_data = batch_data.to(self.device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):
                    task_output, _ = model(data=batch_data)
                    primary_loss_per_node_avg = criterion(task_output, batch_data.y)
                    weight_factor = batch_data.num_nodes / total_nodes_in_level_graph if total_nodes_in_level_graph > 0 else 0.0
                    weighted_primary_loss = primary_loss_per_node_avg * weight_factor
                    l2_reg = sum(p.norm(2).pow(2) for p in model.parameters() if p.requires_grad)
                    loss = weighted_primary_loss + l2_lambda * l2_reg
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            avg_epoch_loss = epoch_loss / len(subgraphs)
            if scheduler:
                scheduler.step(avg_epoch_loss)
            if epoch == epochs: # Only print for the last epoch
                if self.config.DEBUG_VERBOSE:
                    print(f"    Epoch: {epoch:03d}, Avg Batch Loss: {avg_epoch_loss:.4f}")
            if early_stopper and early_stopper.early_stop(avg_epoch_loss):
                print(f"  Early stopping triggered at epoch {epoch}. Best loss: {early_stopper.best_loss:.4f}")
                break

    def _create_clustered_subgraphs(self, graph: DirectedNgramGraph, full_data: Data) -> List[Data]:
        """Partitions the graph into subgraphs, now including the undirected matrix."""
        num_clusters_calculated = math.ceil(graph.number_of_nodes / self.config.GCN_TARGET_NODES_PER_CLUSTER)
        num_clusters = max(self.config.GCN_MIN_CLUSTERS, num_clusters_calculated)
        num_clusters = min(num_clusters, self.config.GCN_MAX_CLUSTERS)
        print(f"  Partitioning graph with {graph.number_of_nodes} nodes into {num_clusters} clusters (target nodes/cluster: {self.config.GCN_TARGET_NODES_PER_CLUSTER})...")

        mathcal_A_combined_sparse_cpu = (graph.mathcal_A_out + graph.mathcal_A_in).coalesce().cpu()
        g_nx = to_networkx(Data(edge_index=mathcal_A_combined_sparse_cpu.indices(), edge_attr=mathcal_A_combined_sparse_cpu.values(), num_nodes=graph.number_of_nodes), to_undirected=True, edge_attrs=['edge_attr'])

        try:
            import metis
            print("  Using METIS for graph partitioning...")
            _, parts = metis.part_graph(g_nx, num_clusters, seed=self.config.RANDOM_STATE)
            partition = {node_idx: part_id for node_idx, part_id in enumerate(parts)}
        except (ImportError, ModuleNotFoundError):
            import community as community_louvain
            print("  METIS not found. Falling back to Louvain for clustering (slower)...")
            partition = community_louvain.best_partition(g_nx, random_state=self.config.RANDOM_STATE)

        clusters = collections.defaultdict(list)
        for node, cluster_id in partition.items():
            clusters[cluster_id].append(node)
        cluster_list = list(clusters.values())
        print(f"  Graph partitioned into {len(cluster_list)} clusters.")

        subgraphs = []
        for cluster_nodes in tqdm(cluster_list, desc="  Creating subgraphs", leave=False):
            cluster_nodes_tensor = torch.tensor(cluster_nodes, dtype=torch.long).to(self.device)

            # Subgraph all three matrices
            sub_edge_index_in, sub_edge_weight_in = subgraph(cluster_nodes_tensor, graph.mathcal_A_in.indices(), graph.mathcal_A_in.values(), relabel_nodes=True, num_nodes=graph.number_of_nodes)
            sub_edge_index_out, sub_edge_weight_out = subgraph(cluster_nodes_tensor, graph.mathcal_A_out.indices(), graph.mathcal_A_out.values(), relabel_nodes=True, num_nodes=graph.number_of_nodes)
            sub_edge_index_undir, sub_edge_weight_undir = subgraph(cluster_nodes_tensor, graph.A_undirected_norm_sparse.indices(), graph.A_undirected_norm_sparse.values(), relabel_nodes=True,
                                                                   num_nodes=graph.number_of_nodes)

            subgraph_data = Data(
                x=full_data.x[cluster_nodes_tensor],
                y=full_data.y[cluster_nodes_tensor] if full_data.y.numel() > 0 else torch.empty(0),
                edge_index_in=sub_edge_index_in, edge_weight_in=sub_edge_weight_in,
                edge_index_out=sub_edge_index_out, edge_weight_out=sub_edge_weight_out,
                edge_index_undirected_norm=sub_edge_index_undir,
                edge_weight_undirected_norm=sub_edge_weight_undir,
                original_indices=cluster_nodes_tensor
            )
            subgraphs.append(subgraph_data)
        return subgraphs

    def _generate_community_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        import networkx as nx
        import community as community_louvain
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), 1
        print(f"  Generating community labels for all {num_nodes} nodes.")
        A_in_w_cpu_sparse = graph.A_in_w.cpu()
        A_out_w_cpu_sparse = graph.A_out_w.cpu()
        combined_adj_sparse_torch = (A_in_w_cpu_sparse + A_out_w_cpu_sparse).coalesce()
        if combined_adj_sparse_torch._nnz() == 0: return torch.zeros(num_nodes, dtype=torch.long), 1
        nx_graph = to_networkx(Data(edge_index=combined_adj_sparse_torch.indices(), edge_attr=combined_adj_sparse_torch.values(), num_nodes=num_nodes), to_undirected=True, edge_attrs=['edge_attr'])
        if nx_graph.number_of_edges() == 0: return torch.zeros(num_nodes, dtype=torch.long), 1
        partition = community_louvain.best_partition(nx_graph, random_state=self.config.RANDOM_STATE, weight='edge_attr')
        labels_list = [partition.get(i, -1) for i in range(num_nodes)]
        unique_labels_from_partition = sorted(list(set(labels_list)))
        if not unique_labels_from_partition or (len(unique_labels_from_partition) == 1 and unique_labels_from_partition[0] == -1):
            return torch.zeros(num_nodes, dtype=torch.long), 1
        else:
            label_map = {lbl: i for i, lbl in enumerate(unique_labels_from_partition)}
            labels = torch.tensor([label_map[lbl] for lbl in labels_list], dtype=torch.long)
            return labels, len(unique_labels_from_partition)

    def _generate_next_node_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), 1
        print(f"  Generating next_node labels for all {num_nodes} nodes.")
        adj_out_weighted_sparse = graph.A_out_w
        labels_list = [-1] * num_nodes
        for i in tqdm(range(num_nodes), desc="  Generating next_node labels", disable=not self.config.DEBUG_VERBOSE):
            row_mask = (adj_out_weighted_sparse.indices()[0] == i)
            if not torch.any(row_mask):
                labels_list[i] = i
            else:
                successors = adj_out_weighted_sparse.indices()[1][row_mask]
                weights = adj_out_weighted_sparse.values()[row_mask]
                max_weight_successors = successors[weights == weights.max()]
                labels_list[i] = random.choice(max_weight_successors.cpu().tolist())
        return torch.tensor(labels_list, dtype=torch.long), num_nodes

    def _generate_closest_amino_acid_labels(self, graph: DirectedNgramGraph, k_hops: int) -> Tuple[torch.Tensor, int]:
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), k_hops + 1
        labels_for_all_nodes = torch.full((num_nodes,), k_hops, dtype=torch.long)
        print(f"  Generating closest_aa labels for all {num_nodes} nodes.")
        adj_out_sparse = graph.A_out_w
        node_sequences = graph.node_sequences
        for start_node_idx in tqdm(range(num_nodes), desc="  Generating closest_aa labels", disable=not self.config.DEBUG_VERBOSE):
            target_aa = random.choice(AMINO_ACID_ALPHABET)
            if target_aa in str(node_sequences[start_node_idx]):
                labels_for_all_nodes[start_node_idx] = 0
            elif k_hops > 0:
                q = collections.deque([(start_node_idx, 0)])
                visited = {start_node_idx}
                found = False
                while q:
                    curr, hop = q.popleft()
                    if hop >= k_hops: continue
                    row_mask = (adj_out_sparse.indices()[0] == curr)
                    if not torch.any(row_mask): continue
                    for neighbor in adj_out_sparse.indices()[1][row_mask]:
                        n_idx = neighbor.item()
                        if n_idx not in visited:
                            visited.add(n_idx)
                            if target_aa in str(node_sequences[n_idx]):
                                labels_for_all_nodes[start_node_idx] = hop + 1
                                found = True
                                break
                            if hop + 1 < k_hops: q.append((n_idx, hop + 1))
                    if found: break
        return labels_for_all_nodes, k_hops + 1

    def run(self):
        DataUtils.print_header("PIPELINE STEP 2: Training ProtGramDirectGCN & Generating Embeddings")
        os.makedirs(self.config.GCN_EMBEDDINGS_DIR, exist_ok=True)
        DataUtils.print_header("Step 1: Loading Protein ID Mapping (if configured)")
        if self.config.ID_MAPPING_MODE != 'none':
            id_mapper_instance = DataLoader(config=self.config)
            self.id_map = id_mapper_instance.generate_id_maps()
            print(f"  Loaded {len(self.id_map)} ID mappings.")
        else:
            self.id_map = {}
        print(f"Using device: {self.device}")
        level_embeddings: Dict[int, np.ndarray] = {}
        level_ngram_to_idx: Dict[int, Dict[str, int]] = {}
        l2_lambda_val = getattr(self.config, 'GCN_L2_REG_LAMBDA', 0.0)

        for n_val in range(1, self.config.GCN_NGRAM_MAX_N + 1):
            DataUtils.print_header(f"Processing N-gram Level: n = {n_val}")
            graph_path = os.path.join(self.config.GRAPH_OBJECTS_DIR, f"ngram_graph_n{n_val}.pkl")
            try:
                graph_obj = DataUtils.load_object(graph_path)
                if not isinstance(graph_obj, DirectedNgramGraph):
                    raise TypeError("Loaded object is not a DirectedNgramGraph")
                graph_obj.n_value = n_val
                if graph_obj.number_of_nodes > 0:
                    graph_obj.A_out_w = graph_obj.A_out_w.to(self.device)
                    graph_obj.A_in_w = graph_obj.A_in_w.to(self.device)
                    graph_obj.A_undirected_norm_sparse = graph_obj.A_undirected_norm_sparse.to(self.device)
                    print("    Re-creating propagation matrices for loaded graph object...")
                    graph_obj._create_propagation_matrices_for_gcn()
            except Exception as e:
                print(f"  ERROR: Could not load or process graph for n={n_val}: {e}")
                continue
            if not graph_obj or graph_obj.number_of_nodes == 0:
                print(f"  Skipping n={n_val} (no nodes or failed to load).")
                level_embeddings[n_val] = np.array([])
                continue
            level_ngram_to_idx[n_val] = graph_obj.node_to_idx
            print(f"  Graph for n={n_val} loaded. Nodes: {graph_obj.number_of_nodes}")
            current_task_type = self.config.GCN_TASK_TYPES_PER_LEVEL.get(n_val, self.config.GCN_DEFAULT_TASK_TYPE)
            print(f"  Selected training task for n={n_val}: '{current_task_type}'")

            num_initial_features: int
            if n_val == 1:
                num_initial_features = self.config.GCN_1GRAM_INIT_DIM
                x = torch.randn(graph_obj.number_of_nodes, num_initial_features, device=self.device)
            else:
                if (n_val - 1) not in level_embeddings or level_embeddings[n_val - 1].size == 0:
                    print(f"  ERROR: Prev level n={n_val - 1} embeddings missing/empty. Skipping.")
                    continue
                prev_level_embeds_np = level_embeddings[n_val - 1]
                prev_level_ngram_to_idx_map = level_ngram_to_idx[n_val - 1]
                num_initial_features = prev_level_embeds_np.shape[1]
                x = torch.zeros(graph_obj.number_of_nodes, num_initial_features, dtype=torch.float, device=self.device)
                print(f"  Initializing features for n={n_val} by pooling (n-1)-gram constituent embeddings...")
                for ngram_str, idx in tqdm(graph_obj.node_to_idx.items(), desc=f"  Initializing n={n_val} features", leave=False, disable=not self.config.DEBUG_VERBOSE):
                    prefix, suffix = ngram_str[:-1], ngram_str[1:]
                    p_idx, s_idx = prev_level_ngram_to_idx_map.get(prefix), prev_level_ngram_to_idx_map.get(suffix)
                    embeds_to_pool = [prev_level_embeds_np[i] for i in [p_idx, s_idx] if i is not None]
                    if embeds_to_pool:
                        x[idx] = torch.from_numpy(np.mean(np.array(embeds_to_pool, dtype=np.float32), axis=0)).to(self.device)
            print(f"  Initial node feature dimension for n={n_val}: {num_initial_features}")

            labels, num_classes = None, 0
            if current_task_type == "community":
                labels, num_classes = self._generate_community_labels(graph_obj)
            elif current_task_type == "next_node":
                labels, num_classes = self._generate_next_node_labels(graph_obj)
            elif current_task_type == "closest_aa":
                labels, num_classes = self._generate_closest_amino_acid_labels(graph_obj, self.config.GCN_CLOSEST_AA_K_HOPS)
            else:
                raise ValueError(f"Unsupported GCN_TASK_TYPE '{current_task_type}' for n={n_val}.")

            full_data = Data(x=x, y=labels.to(self.device))
            full_data.num_nodes = graph_obj.number_of_nodes

            full_layer_dims = [num_initial_features] + self.config.GCN_HIDDEN_LAYER_DIMS
            model = ProtGramDirectGCN(
                layer_dims=full_layer_dims, num_graph_nodes=graph_obj.number_of_nodes,
                task_num_output_classes=num_classes, n_gram_len=n_val,
                one_gram_dim=(self.config.GCN_1GRAM_INIT_DIM if n_val == 1 else 0),
                max_pe_len=self.config.GCN_MAX_PE_LEN, dropout=self.config.GCN_DROPOUT_RATE,
                use_vector_coeffs=self.config.GCN_USE_VECTOR_COEFFS
            )
            optimizer = optim.Adam(model.parameters(), lr=self.config.GCN_LR, weight_decay=self.config.GCN_WEIGHT_DECAY if l2_lambda_val <= 0 else 0.0)

            if self.config.GCN_USE_CLUSTER_TRAINING and graph_obj.number_of_nodes > self.config.GCN_CLUSTER_TRAINING_THRESHOLD_NODES:
                subgraphs = self._create_clustered_subgraphs(graph_obj, full_data)
                self._train_model_clustered(model, subgraphs, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, current_task_type, l2_lambda_val,
                                            total_nodes_in_level_graph=graph_obj.number_of_nodes)
            else:
                # --- MODIFIED: Add all matrices to the full_data object ---
                full_data.edge_index_in = graph_obj.mathcal_A_in.indices().to(self.device)
                full_data.edge_weight_in = graph_obj.mathcal_A_in.values().to(self.device)
                full_data.edge_index_out = graph_obj.mathcal_A_out.indices().to(self.device)
                full_data.edge_weight_out = graph_obj.mathcal_A_out.values().to(self.device)
                full_data.edge_index_undirected_norm = graph_obj.A_undirected_norm_sparse.indices().to(self.device)
                full_data.edge_weight_undirected_norm = graph_obj.A_undirected_norm_sparse.values().to(self.device)
                # --- END MODIFIED ---
                self._train_model_full_batch(model, full_data, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, current_task_type, l2_lambda_val)

            # Ensure data object has all matrices for embedding extraction
            if not hasattr(full_data, 'edge_index_undirected_norm'):
                full_data.edge_index_in = graph_obj.mathcal_A_in.indices().to(self.device)
                full_data.edge_weight_in = graph_obj.mathcal_A_in.values().to(self.device)
                full_data.edge_index_out = graph_obj.mathcal_A_out.indices().to(self.device)
                full_data.edge_weight_out = graph_obj.mathcal_A_out.values().to(self.device)
                full_data.edge_index_undirected_norm = graph_obj.A_undirected_norm_sparse.indices().to(self.device)
                full_data.edge_weight_undirected_norm = graph_obj.A_undirected_norm_sparse.values().to(self.device)

            current_level_embeddings = EmbeddingProcessor.extract_gcn_node_embeddings(model, full_data, self.device)
            level_embeddings[n_val] = current_level_embeddings

            del model, full_data, graph_obj, optimizer, x, labels
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        DataUtils.print_header("Step 3: Pooling N-gram Embeddings to Protein Level")
        final_n_val = self.config.GCN_NGRAM_MAX_N
        if final_n_val not in level_embeddings or level_embeddings[final_n_val].size == 0:
            print(f"ERROR: Final n={final_n_val} embeddings missing/empty. Cannot pool.")
            return
        final_ngram_embeds = level_embeddings[final_n_val]
        final_ngram_map = level_ngram_to_idx[final_n_val]
        protein_sequences = list(DataLoader.parse_sequences(str(self.config.GCN_INPUT_FASTA_PATH)))
        pooled_embeddings = EmbeddingProcessor.pool_ngram_embeddings_for_protein_fast(
            protein_sequences=protein_sequences, n_val=final_n_val,
            ngram_map=final_ngram_map, ngram_embeddings=final_ngram_embeds
        )
        if self.id_map:
            pooled_embeddings = {self.id_map.get(k, k): v for k, v in pooled_embeddings.items()}

        DataUtils.print_header("Step 4: Saving Generated Embeddings")
        output_h5_path = os.path.join(self.config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings.h5")
        with h5py.File(output_h5_path, 'w') as hf:
            for key, vector in tqdm(pooled_embeddings.items(), desc="  Writing H5 File"):
                if vector is not None: hf.create_dataset(key, data=vector)
        print(f"\nSUCCESS: Primary embeddings saved to: {output_h5_path}")

        final_embedding_path_for_sanity_check = output_h5_path

        if self.config.APPLY_PCA_TO_GCN and pooled_embeddings:
            DataUtils.print_header("Step 5: Applying PCA for Dimensionality Reduction")
            pca_embeds = EmbeddingProcessor.apply_pca(pooled_embeddings, self.config.PCA_TARGET_DIMENSION, self.config.RANDOM_STATE)
            if pca_embeds:
                pca_dim = next(iter(pca_embeds.values())).shape[0]
                pca_h5_path = os.path.join(self.config.GCN_EMBEDDINGS_DIR, f"gcn_n{final_n_val}_embeddings_pca{pca_dim}.h5")
                with h5py.File(pca_h5_path, 'w') as hf:
                    for key, vector in tqdm(pca_embeds.items(), desc="  Writing PCA H5 File"):
                        if vector is not None: hf.create_dataset(key, data=vector)
                print(f"  SUCCESS: PCA-reduced embeddings saved to: {pca_h5_path}")
                final_embedding_path_for_sanity_check = pca_h5_path

        if self.config.GCN_RUN_SANITY_CHECK_PPI:
            self._run_sanity_check_ppi(final_embedding_path_for_sanity_check)

        DataUtils.print_header("ProtGramDirectGCN Embedding PIPELINE STEP FINISHED")

    def _run_sanity_check_ppi(self, embedding_path: str):
        # This method remains unchanged from v4.13
        DataUtils.print_header("Step 6: Running Sanity Check PPI Task")
        if not TENSORFLOW_AVAILABLE:
            print("  Skipping sanity check: TensorFlow is not installed.")
            return
        if not os.path.exists(embedding_path):
            print(f"  Skipping sanity check: Embedding file not found at {embedding_path}")
            return
        pos_pairs = GroundTruthLoader.load_interaction_pairs(str(self.config.INTERACTIONS_POSITIVE_PATH), 1)
        neg_pairs = GroundTruthLoader.load_interaction_pairs(str(self.config.INTERACTIONS_NEGATIVE_PATH), 0, sample_n=len(pos_pairs), random_state=self.config.RANDOM_STATE)
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
            train_pairs, test_pairs = train_test_split(pairs_for_eval, test_size=self.config.GCN_SANITY_CHECK_TEST_SPLIT, random_state=self.config.RANDOM_STATE, stratify=labels)
            first_emb_key = next(iter(protein_embeddings.get_keys()))
            embedding_dim = protein_embeddings[first_emb_key].shape[0]
            edge_feature_dim = embedding_dim * 2
            train_gen = partial(EmbeddingProcessor.generate_edge_features_batched, interaction_pairs=train_pairs, protein_embeddings=protein_embeddings, method='concatenate', batch_size=self.config.EVAL_BATCH_SIZE,
                                embedding_dim=embedding_dim)
            test_gen = partial(EmbeddingProcessor.generate_edge_features_batched, interaction_pairs=test_pairs, protein_embeddings=protein_embeddings, method='concatenate', batch_size=self.config.EVAL_BATCH_SIZE,
                               embedding_dim=embedding_dim)
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