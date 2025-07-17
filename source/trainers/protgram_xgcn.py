# G:/My Drive/Knowledge/Research/TWU/Projects/protein-protein interaction prediction/Code/ProtGram-DirectGCN/source/trainers/protgram_xgcn.py

# ==============================================================================
# MODULE: trainers/protgram_xgcn.py
# PURPOSE: Unified trainer for GNNs on ProtGram n-gram graphs with self-supervised tasks.
# VERSION: 2.1 (Restored and integrated full self-supervised label generation logic)
# AUTHOR: Islam Ebeid (Integrated by Coding Partner)
# ==============================================================================

import collections
import gc
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Correctly import community_louvain
import community as community_louvain
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, subgraph
from tqdm import tqdm

from configuration.config import Config
from source.data_builders.graph import DirectedNgramGraph
from source.models.gnn.directgcn import ProtGramDirectGCN
from source.models.gnn.rgcn import RGCN
from source.models.gnn.tongidigcn import TongDiGCN
from source.utils.data import DataUtils, DataLoader, GroundTruthLoader
from source.utils.models import EmbeddingProcessor, EmbeddingLoader

# --- Optional Imports for Sanity Check PPI Task ---
try:
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from source.models.ml.mlp import MLP

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


class ProtGramXGCNTrainer:
    """
    Orchestrates the training of different GNN models on the pre-built
    ProtGram n-gram graphs and generates final protein-level embeddings.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def run(self) -> Dict[str, str]:
        """
        Main execution function. Loops through n-gram levels, trains models,
        and generates final protein embeddings.
        Returns a dictionary of {model_name: output_path}.
        """
        DataUtils.print_header("PIPELINE STEP 2: Training ProtGram Models & Generating Embeddings")

        final_protein_embeddings_per_model = {}
        id_map = self._load_id_map()

        for model_type in self.config.PROTGRAM_MODELS_TO_TRAIN:
            DataUtils.print_header(f"Processing Model Type: {model_type.upper()}")
            ngram_embeddings_per_level = self._train_gnns_hierarchically(model_type)

            final_protein_embeddings = self._pool_to_protein_level(ngram_embeddings_per_level)

            if id_map and final_protein_embeddings:
                # Use the .get() method for safe dictionary access
                final_protein_embeddings = {id_map.get(k, k): v for k, v in final_protein_embeddings.items()}

            final_protein_embeddings_per_model[model_type] = final_protein_embeddings

        output_paths = self._save_final_embeddings(final_protein_embeddings_per_model)

        if self.config.GCN_RUN_SANITY_CHECK_PPI:
            main_model_name = self.config.PROTGRAM_MODELS_TO_TRAIN[0]
            embedding_path_for_check = output_paths.get(f"{main_model_name}_pca", output_paths.get(main_model_name))
            if embedding_path_for_check:
                self._run_sanity_check_ppi(embedding_path_for_check)

        DataUtils.print_header("ProtGram Embedding PIPELINE STEP FINISHED")
        return output_paths

    def _train_gnns_hierarchically(self, model_type: str) -> Dict[int, np.ndarray]:
        ngram_embeddings_per_level: Dict[int, np.ndarray] = {}
        level_ngram_to_idx: Dict[int, Dict[str, int]] = {}
        l2_lambda_val = getattr(self.config, 'GCN_L2_REG_LAMBDA', 0.0)

        for n in range(1, self.config.GCN_NGRAM_MAX_N + 1):
            DataUtils.print_header(f"Processing N-gram Level: n = {n} for model '{model_type}'")

            graph_obj_path = self.config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{n}.pkl"
            if not graph_obj_path.exists():
                print(f"  Graph object not found for n={n}. Skipping.")
                continue

            graph_obj: DirectedNgramGraph = DataUtils.load_object(str(graph_obj_path))
            if graph_obj is None:
                print(f"  Failed to load graph object for n={n}. Skipping.")
                continue

            graph_obj.A_out_w = graph_obj.A_out_w.to(self.device)
            graph_obj.A_in_w = graph_obj.A_in_w.to(self.device)
            graph_obj.A_undirected_norm_sparse = graph_obj.A_undirected_norm_sparse.to(self.device)
            graph_obj._create_propagation_matrices_for_gcn()

            level_ngram_to_idx[n] = graph_obj.node_to_idx
            print(f"  Graph for n={n} loaded. Nodes: {graph_obj.number_of_nodes}")
            if graph_obj.number_of_nodes == 0:
                continue

            if n == 1:
                initial_features = torch.randn(
                    (graph_obj.number_of_nodes, self.config.GCN_1GRAM_INIT_DIM),
                    device=self.device
                )
            else:
                prev_level_embeds = ngram_embeddings_per_level.get(n - 1)
                prev_level_map = level_ngram_to_idx.get(n - 1)
                if prev_level_embeds is None or prev_level_embeds.size == 0 or prev_level_map is None:
                    print(f"  Cannot proceed for n={n}, previous level embeddings not found or empty.")
                    continue
                initial_features = self._pool_lower_level_embeddings(graph_obj, prev_level_embeds, prev_level_map)
                if initial_features is None: continue
                initial_features = initial_features.to(self.device)

            task_type = self.config.GCN_TASK_TYPES_PER_LEVEL.get(n, self.config.GCN_DEFAULT_TASK_TYPE)
            print(f"  Selected self-supervised task for n={n}: '{task_type}'")
            labels, num_classes_for_task = self._generate_task_labels(graph_obj, task_type)

            model = self._build_model(model_type, n, initial_features.shape[1], num_classes_for_task, graph_obj.number_of_nodes)
            if model is None: continue
            model.to(self.device)

            data = self._prepare_data_for_model(model_type, graph_obj, initial_features, labels)
            optimizer = optim.Adam(model.parameters(), lr=self.config.GCN_LR, weight_decay=self.config.GCN_WEIGHT_DECAY if l2_lambda_val <= 0 else 0.0)

            if self.config.GCN_USE_CLUSTER_TRAINING and graph_obj.number_of_nodes > self.config.GCN_CLUSTER_TRAINING_THRESHOLD_NODES:
                subgraphs = self._create_clustered_subgraphs(graph_obj, data)
                self._train_model_clustered(model, subgraphs, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, task_type, l2_lambda_val,
                                            total_nodes_in_level_graph=graph_obj.number_of_nodes)
            else:
                self._train_model_full_batch(model, data, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, task_type, l2_lambda_val)

            ngram_embeddings_per_level[n] = EmbeddingProcessor.extract_gcn_node_embeddings(
                model, data, graph_obj, self.config, self.device, self._create_clustered_subgraphs
            )
            print(f"  Generated {ngram_embeddings_per_level[n].shape[0]} embeddings of dim {ngram_embeddings_per_level[n].shape[1]} for n={n}.")
            del model, data, graph_obj, initial_features, labels, optimizer
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        return ngram_embeddings_per_level

    def _build_model(self, model_type: str, n_val: int, in_channels: int, num_classes: int, num_nodes: int):
        """Model factory for creating different GNN architectures."""
        layer_dims = [in_channels] + self.config.GCN_HIDDEN_LAYER_DIMS
        if num_classes <= 0: num_classes = 1

        if model_type == 'directgcn':
            return ProtGramDirectGCN(
                layer_dims=layer_dims, num_graph_nodes=num_nodes,
                task_num_output_classes=num_classes, n_gram_len=n_val,
                one_gram_dim=self.config.GCN_1GRAM_INIT_DIM, max_pe_len=self.config.GCN_MAX_PE_LEN,
                dropout=self.config.GCN_DROPOUT_RATE, use_vector_coeffs=self.config.GCN_USE_VECTOR_COEFFS
            )
        elif model_type == 'rgcn':
            return RGCN(in_channels, self.config.GCN_HIDDEN_LAYER_DIMS[-1], num_classes, num_relations=2)
        elif model_type == 'tongdigcn':
            return TongDiGCN(in_channels, self.config.GCN_HIDDEN_LAYER_DIMS[0], num_classes)
        else:
            print(f"  ERROR: Unknown model type '{model_type}' for ProtGram training.")
            return None

    def _prepare_data_for_model(self, model_type: str, graph: DirectedNgramGraph, features: torch.Tensor, labels: torch.Tensor) -> Data:
        """Prepares a PyG Data object tailored to the specific model's needs."""
        data_dict = {'x': features, 'y': labels.to(self.device)}

        if model_type == 'directgcn':
            data_dict.update({
                'edge_index_in': graph.mathcal_A_in.indices(), 'edge_weight_in': graph.mathcal_A_in.values(),
                'edge_index_out': graph.mathcal_A_out.indices(), 'edge_weight_out': graph.mathcal_A_out.values(),
                'edge_index_undirected_norm': graph.A_undirected_norm_sparse.indices(),
                'edge_weight_undirected_norm': graph.A_undirected_norm_sparse.values()
            })
        elif model_type == 'rgcn':
            edge_index_out = graph.A_out_w.indices()
            edge_index_in = graph.A_in_w.indices()
            edge_type_out = torch.zeros(edge_index_out.size(1), dtype=torch.long, device=self.device)
            edge_type_in = torch.ones(edge_index_in.size(1), dtype=torch.long, device=self.device)
            data_dict['edge_index'] = torch.cat([edge_index_out, edge_index_in], dim=1)
            data_dict['edge_type'] = torch.cat([edge_type_out, edge_type_in], dim=0)
        elif model_type == 'tongdigcn':
            # This model expects forward and backward edges. We use the raw weighted matrices.
            data_dict['edge_index'] = graph.A_out_w.indices()
            data_dict['edge_index_backward'] = graph.A_in_w.indices()
        else:
            raise ValueError(f"Cannot prepare data for unknown model type: {model_type}")
        return Data.from_dict(data_dict)

    def _train_model_full_batch(self, model: nn.Module, data: Data, optimizer: torch.optim.Optimizer, epochs: int,
                                task_type: str, l2_lambda: float = 0.0):
        """Full-batch training logic."""
        model.train()
        model.to(self.device)
        data = data.to(self.device)
        scheduler = None
        if self.config.GCN_USE_LR_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.GCN_LR_SCHEDULER_PATIENCE, factor=self.config.GCN_LR_SCHEDULER_FACTOR)
        early_stopper = None
        if self.config.GCN_USE_EARLY_STOPPING:
            early_stopper = EarlyStopper(patience=self.config.GCN_EARLY_STOPPING_PATIENCE, min_delta=self.config.GCN_EARLY_STOPPING_MIN_DELTA)
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        criterion = F.nll_loss
        print(f"  Starting full-batch training for up to {epochs} epochs (Task: {task_type}, L2 lambda: {l2_lambda})...")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                task_output, _ = model(data=data)
                primary_loss = criterion(task_output, data.y)
                l2_reg = sum(p.norm(2).pow(2) for p in model.parameters() if p.requires_grad)
                loss = primary_loss + l2_lambda * l2_reg
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step(loss)
            if self.config.DEBUG_VERBOSE and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"    Epoch: {epoch:03d}, Total Loss: {loss.item():.4f}, Primary Loss: {primary_loss.item():.4f}, L2: {(l2_lambda * l2_reg).item():.4f}")
            if early_stopper and early_stopper.early_stop(loss.item()):
                print(f"  Early stopping triggered at epoch {epoch}. Best loss: {early_stopper.best_loss:.4f}")
                break

    def _train_model_clustered(self, model: nn.Module, subgraphs: List[Data], optimizer: torch.optim.Optimizer,
                               epochs: int, task_type: str, l2_lambda: float = 0.0,
                               total_nodes_in_level_graph: int = 1):
        """Clustered training logic."""
        model.train()
        model.to(self.device)
        scheduler = None
        if self.config.GCN_USE_LR_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.GCN_LR_SCHEDULER_PATIENCE, factor=self.config.GCN_LR_SCHEDULER_FACTOR)
        early_stopper = None
        if self.config.GCN_USE_EARLY_STOPPING:
            early_stopper = EarlyStopper(patience=self.config.GCN_EARLY_STOPPING_PATIENCE, min_delta=self.config.GCN_EARLY_STOPPING_MIN_DELTA)
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        criterion = F.nll_loss
        print(f"  Starting Cluster-GCN style training for up to {epochs} epochs on {len(subgraphs)} subgraphs (Task: {task_type})...")
        for epoch in range(1, epochs + 1):
            random.shuffle(subgraphs)
            epoch_loss = 0.0
            for batch_data in tqdm(subgraphs, desc=f"  Epoch {epoch}", leave=False, disable=not self.config.DEBUG_VERBOSE):
                batch_data = batch_data.to(self.device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
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
            avg_epoch_loss = epoch_loss / len(subgraphs) if subgraphs else 0
            if self.config.DEBUG_VERBOSE and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"    Epoch: {epoch:03d}, Avg Batch Loss: {avg_epoch_loss:.4f}")
            if scheduler:
                scheduler.step(avg_epoch_loss)
            if early_stopper and early_stopper.early_stop(avg_epoch_loss):
                print(f"  Early stopping triggered at epoch {epoch}. Best loss: {early_stopper.best_loss:.4f}")
                break

    def _create_clustered_subgraphs(self, graph: DirectedNgramGraph, full_data: Data) -> List[Data]:
        """Partitions the graph into subgraphs, including all necessary matrices."""
        num_clusters_calculated = math.ceil(graph.number_of_nodes / self.config.GCN_TARGET_NODES_PER_CLUSTER)
        num_clusters = max(self.config.GCN_MIN_CLUSTERS, num_clusters_calculated)
        num_clusters = min(num_clusters, self.config.GCN_MAX_CLUSTERS, graph.number_of_nodes)
        print(f"  Partitioning graph with {graph.number_of_nodes} nodes into {num_clusters} clusters (target nodes/cluster: {self.config.GCN_TARGET_NODES_PER_CLUSTER})...")

        A_combined_cpu = (graph.A_in_w.cpu() + graph.A_out_w.cpu()).coalesce()
        g_nx = to_networkx(Data(edge_index=A_combined_cpu.indices(), edge_attr=A_combined_cpu.values(), num_nodes=graph.number_of_nodes), to_undirected=True, edge_attrs=['edge_attr'])

        try:
            import metis
            print("  Using METIS for graph partitioning...")
            _, parts = metis.part_graph(g_nx, num_clusters, seed=self.config.RANDOM_STATE)
            partition = {node_idx: part_id for node_idx, part_id in enumerate(parts)}
        except (ImportError, ModuleNotFoundError):
            print("  METIS not found. Falling back to Louvain for clustering (slower)...")
            partition = community_louvain.best_partition(g_nx, random_state=self.config.RANDOM_STATE, weight='edge_attr')

        clusters = collections.defaultdict(list)
        for node, cluster_id in partition.items():
            clusters[cluster_id].append(node)
        cluster_list = list(clusters.values())
        print(f"  Graph partitioned into {len(cluster_list)} clusters.")

        subgraphs = []
        for cluster_nodes in tqdm(cluster_list, desc="  Creating subgraphs", leave=False):
            nodes_tensor_cpu = torch.tensor(cluster_nodes, dtype=torch.long, device='cpu')

            # Create subgraph for all necessary matrices
            sub_x = full_data.x[nodes_tensor_cpu]
            sub_y = full_data.y[nodes_tensor_cpu] if full_data.y.numel() > 0 else torch.empty(0, dtype=torch.long)

            subgraph_data = Data(x=sub_x, y=sub_y, original_indices=nodes_tensor_cpu)

            # Subgraph the matrices required by the models being used.
            # This is more efficient than subgraphing everything every time.
            for model_type in self.config.PROTGRAM_MODELS_TO_TRAIN:
                if model_type == 'directgcn':
                    sub_edge_index_in, sub_edge_weight_in = subgraph(nodes_tensor_cpu, graph.mathcal_A_in.indices(), graph.mathcal_A_in.values(), relabel_nodes=True, num_nodes=graph.number_of_nodes)
                    sub_edge_index_out, sub_edge_weight_out = subgraph(nodes_tensor_cpu, graph.mathcal_A_out.indices(), graph.mathcal_A_out.values(), relabel_nodes=True, num_nodes=graph.number_of_nodes)
                    sub_edge_index_undir, sub_edge_weight_undir = subgraph(nodes_tensor_cpu, graph.A_undirected_norm_sparse.indices(), graph.A_undirected_norm_sparse.values(), relabel_nodes=True,
                                                                           num_nodes=graph.number_of_nodes)
                    subgraph_data.edge_index_in, subgraph_data.edge_weight_in = sub_edge_index_in, sub_edge_weight_in
                    subgraph_data.edge_index_out, subgraph_data.edge_weight_out = sub_edge_index_out, sub_edge_weight_out
                    subgraph_data.edge_index_undirected_norm, subgraph_data.edge_weight_undirected_norm = sub_edge_index_undir, sub_edge_weight_undir

                elif model_type == 'tongdigcn':
                    sub_edge_index_fwd, _ = subgraph(nodes_tensor_cpu, graph.A_out_w.indices(), relabel_nodes=True, num_nodes=graph.number_of_nodes)
                    sub_edge_index_bwd, _ = subgraph(nodes_tensor_cpu, graph.A_in_w.indices(), relabel_nodes=True, num_nodes=graph.number_of_nodes)
                    subgraph_data.edge_index = sub_edge_index_fwd
                    subgraph_data.edge_index_backward = sub_edge_index_bwd
            subgraphs.append(subgraph_data)

        return subgraphs

    def _generate_task_labels(self, graph: DirectedNgramGraph, task_type: str) -> Tuple[Optional[torch.Tensor], int]:
        """
        Dispatcher for generating labels for different self-supervised tasks.
        """
        if task_type == 'community':
            return self._generate_community_labels(graph)
        elif task_type == 'next_node':
            return self._generate_next_node_labels(graph)
        elif task_type == 'closest_aa':
            return self._generate_closest_amino_acid_labels(graph, self.config.GCN_CLOSEST_AA_K_HOPS)
        else:
            raise ValueError(f"Unknown self-supervised task type: '{task_type}'")

    def _generate_community_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        """Generates community detection labels using the Louvain algorithm."""
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), 1
        print(f"  Generating community labels for all {num_nodes} nodes.")

        # Use the undirected, normalized adjacency matrix as it represents the core structure
        # for community detection better than directed, weighted matrices.
        coo = graph.A_undirected_norm_sparse.cpu().coalesce()
        if coo._nnz() == 0: return torch.zeros(num_nodes, dtype=torch.long), 1

        # Convert to NetworkX graph
        nx_graph = to_networkx(Data(edge_index=coo.indices(), edge_attr=coo.values(), num_nodes=num_nodes), to_undirected=True, edge_attrs=['edge_attr'])
        if nx_graph.number_of_edges() == 0: return torch.zeros(num_nodes, dtype=torch.long), 1

        partition = community_louvain.best_partition(nx_graph, random_state=self.config.RANDOM_STATE, weight='edge_attr')
        labels_list = [partition.get(i, -1) for i in range(num_nodes)]

        # Map labels to be contiguous from 0
        unique_labels = sorted(list(set(labels_list)))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        labels = torch.tensor([label_map[lbl] for lbl in labels_list], dtype=torch.long)

        num_classes = len(unique_labels)
        print(f"    Found {num_classes} communities.")
        return labels, num_classes

    def _generate_next_node_labels(self, graph: DirectedNgramGraph) -> Tuple[torch.Tensor, int]:
        """Generates labels by predicting the most likely next node based on transition weights."""
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), 1
        print(f"  Generating next_node labels for all {num_nodes} nodes.")
        adj_out_weighted_sparse = graph.A_out_w
        labels_list = [-1] * num_nodes
        for i in tqdm(range(num_nodes), desc="  Generating next_node labels", disable=not self.config.DEBUG_VERBOSE):
            row_mask = (adj_out_weighted_sparse.indices()[0] == i)
            if not torch.any(row_mask):
                labels_list[i] = i  # Self-loop if no outgoing edges
            else:
                successors = adj_out_weighted_sparse.indices()[1][row_mask]
                weights = adj_out_weighted_sparse.values()[row_mask]
                max_weight_successors = successors[weights == weights.max()]
                labels_list[i] = random.choice(max_weight_successors.cpu().tolist())
        return torch.tensor(labels_list, dtype=torch.long), num_nodes

    def _generate_closest_amino_acid_labels(self, graph: DirectedNgramGraph, k_hops: int) -> Tuple[torch.Tensor, int]:
        """Generates labels by finding the shortest path distance to a randomly chosen amino acid."""
        num_nodes = graph.number_of_nodes
        if num_nodes == 0: return torch.empty(0, dtype=torch.long), k_hops + 1
        labels = torch.full((num_nodes,), k_hops, dtype=torch.long)
        print(f"  Generating closest_aa labels for all {num_nodes} nodes (k={k_hops}).")
        adj_out = graph.A_out_w.cpu()
        node_sequences = graph.node_sequences
        for start_node in tqdm(range(num_nodes), desc="  Generating closest_aa labels", disable=not self.config.DEBUG_VERBOSE):
            target_aa = random.choice(AMINO_ACID_ALPHABET)
            if target_aa in str(node_sequences[start_node]):
                labels[start_node] = 0
                continue
            q = collections.deque([(start_node, 0)])
            visited = {start_node}
            found_at_hop = -1
            while q:
                curr, hop = q.popleft()
                if hop >= k_hops: break
                row_mask = (adj_out.indices()[0] == curr)
                for neighbor in adj_out.indices()[1][row_mask]:
                    n_idx = neighbor.item()
                    if n_idx not in visited:
                        visited.add(n_idx)
                        if target_aa in str(node_sequences[n_idx]):
                            found_at_hop = hop + 1
                            break
                        q.append((n_idx, hop + 1))
                if found_at_hop != -1:
                    break
            if found_at_hop != -1:
                labels[start_node] = found_at_hop
        return labels, k_hops + 1

    def _load_id_map(self):
        """Loads the UniProt ID mapping file if configured."""
        DataUtils.print_header("Step 1: Loading Protein ID Mapping (if configured)")
        if getattr(self.config, 'ID_MAPPING_MODE', 'none') != 'none':
            id_mapper_instance = DataLoader(config=self.config)
            id_map_result = id_mapper_instance.generate_id_maps()
            print(f"  ID mapping result of type '{type(id_map_result)}' loaded.")
            return id_map_result
        return {}

    @staticmethod
    def _pool_lower_level_embeddings(graph_obj: DirectedNgramGraph, prev_level_embeddings: np.ndarray, prev_level_map: Dict[str, int]) -> Optional[torch.Tensor]:
        """Pools embeddings from level n-1 to initialize features for level n by concatenating prefix and suffix embeddings."""
        print(f"  Initializing features for n={graph_obj.n_value} by pooling (n-1)-gram constituent embeddings...")
        num_current_nodes = graph_obj.number_of_nodes
        prev_embedding_dim = prev_level_embeddings.shape[1]
        # FIX: The new dimension should be sum, not product, if we are concatenating.
        new_features = torch.zeros((num_current_nodes, prev_embedding_dim * 2), dtype=torch.float32)

        for i, current_ngram in tqdm(enumerate(graph_obj.node_sequences), total=num_current_nodes, desc=f"  Initializing n={graph_obj.n_value} features", leave=False):
            prefix, suffix = current_ngram[:-1], current_ngram[1:]
            prefix_idx, suffix_idx = prev_level_map.get(prefix), prev_level_map.get(suffix)

            prefix_emb = torch.from_numpy(prev_level_embeddings[prefix_idx]) if prefix_idx is not None else torch.zeros(prev_embedding_dim)
            suffix_emb = torch.from_numpy(prev_level_embeddings[suffix_idx]) if suffix_idx is not None else torch.zeros(prev_embedding_dim)

            new_features[i] = torch.cat([prefix_emb, suffix_emb])
        return new_features

    def _pool_to_protein_level(self, ngram_embeddings: Dict[int, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """Pools the final n-gram embeddings to the protein level."""
        DataUtils.print_header("Step 3: Pooling N-gram Embeddings to Protein Level")
        final_n = self.config.GCN_NGRAM_MAX_N
        final_ngram_embeddings = ngram_embeddings.get(final_n)

        if final_ngram_embeddings is None or final_ngram_embeddings.size == 0:
            print(f"  ERROR: No n-gram embeddings found for the final level (n={final_n}). Cannot generate protein embeddings.")
            return {}

        graph_obj_path = self.config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{final_n}.pkl"
        graph_obj: DirectedNgramGraph = DataUtils.load_object(str(graph_obj_path))
        if graph_obj is None:
            print(f"  ERROR: Failed to load graph object for n={final_n} for pooling.")
            return {}
        ngram_map = graph_obj.node_to_idx
        del graph_obj

        protein_sequences = list(DataLoader.parse_sequences([str(p) for p in self.config.SEQUENCE_FILE_PATHS]))
        pooled_embeddings = EmbeddingProcessor.pool_ngram_embeddings_for_protein_fast(
            protein_sequences=protein_sequences, n_val=final_n,
            ngram_map=ngram_map, ngram_embeddings=final_ngram_embeddings
        )
        return pooled_embeddings

    def _save_final_embeddings(self, final_embeddings_per_model: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, str]:
        """Saves the final generated protein embeddings to H5 files."""
        DataUtils.print_header("Step 4: Saving Generated Embeddings")
        output_paths = {}

        for model_type, protein_embeddings in final_embeddings_per_model.items():
            if not protein_embeddings:
                print(f"  No embeddings to save for model type '{model_type}'.")
                continue

            base_filename = f"{model_type}_protgram_n{self.config.GCN_NGRAM_MAX_N}_embeddings.h5"
            output_path = self.config.RESULTS_GCN_EMBEDDINGS_DIR / base_filename
            self._write_h5(protein_embeddings, output_path, f"Writing H5 File for {model_type}")
            print(f"\nSUCCESS: Primary embeddings for '{model_type}' saved to: {output_path}")
            output_paths[model_type] = str(output_path)

            if self.config.APPLY_PCA_TO_GCN:
                DataUtils.print_header(f"Step 5: Applying PCA for '{model_type}'")
                pca_embeddings = EmbeddingProcessor.apply_pca(protein_embeddings, self.config.PCA_TARGET_DIMENSION, self.config.RANDOM_STATE)
                if pca_embeddings:
                    pca_filename = f"{model_type}_protgram_n{self.config.GCN_NGRAM_MAX_N}_embeddings_pca{self.config.PCA_TARGET_DIMENSION}.h5"
                    pca_output_path = self.config.RESULTS_GCN_EMBEDDINGS_DIR / pca_filename
                    self._write_h5(pca_embeddings, pca_output_path, f"Writing PCA H5 for {model_type}")
                    print(f"\nSUCCESS: PCA-reduced embeddings for '{model_type}' saved to: {pca_output_path}")
                    output_paths[f"{model_type}_pca"] = str(pca_output_path)
        return output_paths

    @staticmethod
    def _write_h5(embeddings_dict: Dict, path: Path, desc: str):
        """Helper function to write a dictionary of embeddings to an HDF5 file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, 'w') as hf:
            for key, value in tqdm(embeddings_dict.items(), desc=f"  {desc}"):
                if value is not None: hf.create_dataset(key, data=value)

    def _run_sanity_check_ppi(self, embedding_path: str):
        """Performs a quick PPI link prediction task to validate the generated embeddings."""
        DataUtils.print_header("Step 6: Running Sanity Check PPI Task")
        if not TENSORFLOW_AVAILABLE:
            print("  Skipping sanity check: TensorFlow is not installed.")
            return
        if not os.path.exists(embedding_path):
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
            train_pairs, test_pairs = train_test_split(pairs_for_eval, test_size=self.config.GCN_SANITY_CHECK_TEST_SPLIT, random_state=self.config.RANDOM_STATE, stratify=labels)

            first_emb_key = next(iter(protein_embeddings.get_keys()))
            embedding_dim = protein_embeddings[first_emb_key].shape[0]
            edge_feature_dim = embedding_dim * 2

            train_gen = partial(EmbeddingProcessor.generate_edge_features_batched, interaction_pairs=train_pairs, protein_embeddings=protein_embeddings, method='concatenate', batch_size=self.config.EVAL_BATCH_SIZE,
                                embedding_dim=embedding_dim)
            test_gen = partial(EmbeddingProcessor.generate_edge_features_batched, interaction_pairs=test_pairs, protein_embeddings=protein_embeddings, method='concatenate', batch_size=self.config.EVAL_BATCH_SIZE,
                               embedding_dim=embedding_dim)

            output_sig = (tf.TensorSpec(shape=(None, edge_feature_dim), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.int32))
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
