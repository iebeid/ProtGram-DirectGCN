# ==============================================================================
# MODULE: trainers/protgram_xgcn.py
# PURPOSE: Unified trainer for GNNs on ProtGram n-gram graphs.
# VERSION: 6.0 (Corrected clustered training logic to prevent GPU memory errors)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import collections
import gc
import math
import random
from contextlib import nullcontext
from typing import Dict, Optional, List, Mapping

import community as community_louvain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx
from tqdm.auto import tqdm

from configuration.config import Config
from source.data_builders.graph import DirectedNgramGraph
from source.data_builders.xgcn import XGCNDataset
from source.models.gnn.spectral.directgcn import DirectGCN
from source.models.gnn.spectral.rgcn import RGCN
from source.models.gnn.spectral.tongidigcn import TongDiGCN
from source.utils.data import DataUtils, IDMapGenerator
from source.utils.models import EmbeddingProcessor, EarlyStopper
from source.utils.post import PostUtils


class ProtGramXGCNTrainer:
    """
    Orchestrates the training of different GNN models on the pre-built
    ProtGram n-gram graphs and generates final protein-level embeddings.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_generator = XGCNDataset(config)
        self.helpers = PostUtils(config)
        print(f"Using device: {self.device}")

    def run(self) -> Dict[str, str]:
        """
        Main execution function. Loops through n-gram levels, trains models,
        and generates final protein-level embeddings.
        """
        DataUtils.print_header("PIPELINE STEP: Training ProtGram Models & Generating Embeddings")

        final_protein_embeddings_per_model = {}
        id_map = self._load_id_map()

        context = id_map if isinstance(id_map, IDMapGenerator) else nullcontext(id_map)

        with context as mapper:
            for model_type in self.config.PROTGRAM_MODELS_TO_TRAIN:
                DataUtils.print_header(f"Processing Model Type: {model_type.upper()}")
                ngram_embeddings_per_level = self._train_gnns_hierarchically(model_type)

                final_protein_embeddings = self.helpers.pool_to_protein_level(ngram_embeddings_per_level)

                if mapper and final_protein_embeddings:
                    print("  Applying ID mapping to final protein embeddings...")
                    final_protein_embeddings = {mapper.get(k, k): v for k, v in final_protein_embeddings.items()}

                final_protein_embeddings_per_model[model_type] = final_protein_embeddings

        output_paths = self.helpers.save_final_embeddings(final_protein_embeddings_per_model)

        if self.config.GCN_RUN_SANITY_CHECK_PPI:
            main_model_name = self.config.PROTGRAM_MODELS_TO_TRAIN[0]
            embedding_path_for_check = output_paths.get(f"{main_model_name}_pca", output_paths.get(main_model_name))
            if embedding_path_for_check:
                self.helpers.run_sanity_check_ppi(embedding_path_for_check)

        DataUtils.print_header("ProtGram Embedding PIPELINE STEP FINISHED")
        return output_paths

    def _load_graph_for_level(self, n: int) -> Optional[DirectedNgramGraph]:
        """Loads the graph object for a specific n-gram level and prepares it for the GPU."""
        graph_obj_path = self.config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{n}.pkl"
        if not graph_obj_path.exists():
            print(f"  Graph object not found for n={n}. Skipping.")
            return None

        graph_obj: DirectedNgramGraph = DataUtils.load_object(str(graph_obj_path))
        if graph_obj is None or graph_obj.number_of_nodes == 0:
            print(f"  Failed to load graph object or graph is empty for n={n}. Skipping.")
            return None

        print(f"  Graph for n={n} loaded. Nodes: {graph_obj.number_of_nodes}")
        # Keep adjacency matrices on CPU for subgraph creation, move to GPU inside the training loop
        graph_obj.A_out_w = graph_obj.A_out_w.cpu()
        graph_obj.A_in_w = graph_obj.A_in_w.cpu()
        graph_obj.A_undirected_norm_sparse = graph_obj.A_undirected_norm_sparse.cpu()
        graph_obj._create_propagation_matrices_for_gcn()
        return graph_obj

    def _get_initial_features_for_level(self, n: int, graph_obj: DirectedNgramGraph,
                                        prev_level_embeddings: Optional[np.ndarray],
                                        prev_level_map: Optional[Dict[str, int]]) -> Optional[torch.Tensor]:
        """Generates the initial node features for the current n-gram level."""
        if n == 1:
            # Keep features on CPU initially
            return torch.randn((graph_obj.number_of_nodes, self.config.GCN_1GRAM_INIT_DIM))
        else:
            if prev_level_embeddings is None or prev_level_embeddings.size == 0 or prev_level_map is None:
                print(f"  Cannot proceed for n={n}, previous level embeddings not found or empty.")
                return None
            initial_features = self.helpers.pool_lower_level_embeddings(graph_obj, prev_level_embeddings, prev_level_map)
            return initial_features if initial_features is not None else None

    def _train_gnns_hierarchically(self, model_type: str) -> Dict[int, np.ndarray]:
        """The main hierarchical training loop, now refactored to use helper methods."""
        ngram_embeddings_per_level: Dict[int, np.ndarray] = {}
        level_ngram_to_idx: Dict[int, Dict[str, int]] = {}

        for n in range(1, self.config.GCN_NGRAM_MAX_N + 1):
            DataUtils.print_header(f"Processing N-gram Level: n = {n} for model '{model_type}'")

            graph_obj = self._load_graph_for_level(n)
            if not graph_obj: continue

            level_ngram_to_idx[n] = graph_obj.node_to_idx
            prev_embeds = ngram_embeddings_per_level.get(n - 1)
            prev_map = level_ngram_to_idx.get(n - 1)

            initial_features = self._get_initial_features_for_level(n, graph_obj, prev_embeds, prev_map)
            if initial_features is None: continue

            task_type = self.config.GCN_TASK_TYPES_PER_LEVEL.get(n, self.config.GCN_DEFAULT_TASK_TYPE)
            labels, num_classes_for_task = self.label_generator.generate_task_labels(graph_obj, task_type)

            model = self._build_model(model_type, n, initial_features.shape[1], num_classes_for_task, graph_obj.number_of_nodes)
            if model is None: continue

            # The 'data' object now holds all features and labels on the CPU. It will be moved to the GPU in batches/subgraphs.
            data = Data(x=initial_features, y=labels, graph_obj=graph_obj)

            optimizer = optim.Adam(model.parameters(), lr=self.config.GCN_LR, weight_decay=self.config.GCN_WEIGHT_DECAY if self.config.GCN_L2_REG_LAMBDA <= 0 else 0.0)

            self._train_single_level(model, graph_obj, data, optimizer, model_type)

            ngram_embeddings_per_level[n] = EmbeddingProcessor.extract_gcn_node_embeddings(
                model, data, graph_obj, self.config, self.device,
                lambda g, d: self._partition_graph(g)
            )

            print(f"  Generated {ngram_embeddings_per_level[n].shape[0]} embeddings of dim {ngram_embeddings_per_level[n].shape[1]} for n={n}.")
            del model, data, graph_obj, initial_features, labels, optimizer
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        return ngram_embeddings_per_level

    def _build_model(self, model_type: str, n_val: int, in_channels: int, num_classes: int, num_nodes: int) -> Optional[nn.Module]:
        """Model factory for creating different GNN architectures."""
        layer_dims = [in_channels] + self.config.GCN_HIDDEN_LAYER_DIMS
        if num_classes <= 0:
            num_classes = 1

        if model_type == 'directgcn':
            return DirectGCN(
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

    def _train_single_level(self, model: nn.Module, graph_obj: DirectedNgramGraph, data: Data, optimizer: torch.optim.Optimizer, model_type: str):
        """Orchestrates the training for a single level, choosing between full-batch and clustered training."""
        l2_lambda_val = getattr(self.config, 'GCN_L2_REG_LAMBDA', 0.0)
        task_type = self.config.GCN_TASK_TYPES_PER_LEVEL.get(graph_obj.n_value, self.config.GCN_DEFAULT_TASK_TYPE)

        if self.config.GCN_USE_CLUSTER_TRAINING and graph_obj.number_of_nodes > self.config.GCN_CLUSTER_TRAINING_THRESHOLD_NODES:
            node_partitions = self._partition_graph(graph_obj)
            self._train_single_level_clustered(model, data, node_partitions, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, task_type, l2_lambda_val)
        else:
            self._train_single_level_full_batch(model, data, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, task_type, l2_lambda_val)

    def _train_single_level_full_batch(self, model: nn.Module, data: Data, optimizer: torch.optim.Optimizer, epochs: int,
                                       task_type: str, l2_lambda: float = 0.0):
        """Full-batch training logic for a single GNN level."""
        model.train()
        model.to(self.device)
        # For full batch, we prepare the data once and move it to the device
        full_data_gpu = self._prepare_data_for_model(model.__class__.__name__.lower(), data.graph_obj, data.x, data.y).to(self.device)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.GCN_LR_SCHEDULER_PATIENCE, factor=self.config.GCN_LR_SCHEDULER_FACTOR) if self.config.GCN_USE_LR_SCHEDULER else None
        early_stopper = EarlyStopper(patience=self.config.GCN_EARLY_STOPPING_PATIENCE, min_delta=self.config.GCN_EARLY_STOPPING_MIN_DELTA) if self.config.GCN_USE_EARLY_STOPPING else None
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        criterion = F.cross_entropy
        print(f"  Starting full-batch training for up to {epochs} epochs (Task: {task_type}, L2 lambda: {l2_lambda})...")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                output, _ = model(data=full_data_gpu)
                primary_loss = criterion(output, full_data_gpu.y)
                l2_reg = sum(p.norm(2).pow(2) for p in model.parameters() if p.requires_grad)
                loss = primary_loss + l2_lambda * l2_reg
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler: scheduler.step(loss)
            if self.config.DEBUG_VERBOSE and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"    Epoch: {epoch:03d}, Total Loss: {loss.item():.4f}, Primary Loss: {primary_loss.item():.4f}, L2: {(l2_lambda * l2_reg).item():.4f}")
            if early_stopper and early_stopper.early_stop(loss.item()):
                print(f"  Early stopping triggered at epoch {epoch}. Best loss: {early_stopper.best_loss:.4f}")
                break

    def _train_single_level_clustered(self, model: nn.Module, full_data: Data, node_partitions: List[List[int]],
                                      optimizer: torch.optim.Optimizer, epochs: int, task_type: str, l2_lambda: float = 0.0):
        """Clustered training logic for a single GNN level."""
        model.train()
        model.to(self.device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.GCN_LR_SCHEDULER_PATIENCE, factor=self.config.GCN_LR_SCHEDULER_FACTOR) if self.config.GCN_USE_LR_SCHEDULER else None
        early_stopper = EarlyStopper(patience=self.config.GCN_EARLY_STOPPING_PATIENCE, min_delta=self.config.GCN_EARLY_STOPPING_MIN_DELTA) if self.config.GCN_USE_EARLY_STOPPING else None
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        criterion = F.cross_entropy
        print(f"  Starting Cluster-GCN style training for up to {epochs} epochs on {len(node_partitions)} subgraphs (Task: {task_type})...")

        for epoch in range(1, epochs + 1):
            random.shuffle(node_partitions)
            epoch_loss = 0.0
            for node_idx_batch in tqdm(node_partitions, desc=f"  Epoch {epoch}", leave=False, disable=not self.config.DEBUG_VERBOSE):
                # --- FIX: Create a self-contained subgraph for each batch ---
                subgraph_data = full_data.graph_obj.create_subgraph_data_for_model(
                    model_type=model.__class__.__name__.lower(),
                    full_features=full_data.x,
                    full_labels=full_data.y,
                    node_subset=torch.tensor(node_idx_batch, dtype=torch.long)
                ).to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    output, _ = model(data=subgraph_data)
                    primary_loss = criterion(output, subgraph_data.y)
                    l2_reg = sum(p.norm(2).pow(2) for p in model.parameters() if p.requires_grad)
                    loss = primary_loss + l2_lambda * l2_reg
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(node_partitions) if node_partitions else 0
            if self.config.DEBUG_VERBOSE and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"    Epoch: {epoch:03d}, Avg Batch Loss: {avg_epoch_loss:.4f}")
            if scheduler: scheduler.step(avg_epoch_loss)
            if early_stopper and early_stopper.early_stop(avg_epoch_loss):
                print(f"  Early stopping triggered at epoch {epoch}. Best loss: {early_stopper.best_loss:.4f}")
                break

    def _partition_graph(self, graph: DirectedNgramGraph) -> List[List[int]]:
        """
        Partitions the graph into clusters of nodes for batch training.
        This version is now model-agnostic and only returns the node indices for each partition.
        """
        if graph.number_of_nodes == 0: return []
        num_clusters_calculated = math.ceil(graph.number_of_nodes / self.config.GCN_TARGET_NODES_PER_CLUSTER)
        num_clusters = max(self.config.GCN_MIN_CLUSTERS, num_clusters_calculated)
        num_clusters = min(num_clusters, self.config.GCN_MAX_CLUSTERS, graph.number_of_nodes)
        print(f"  Partitioning graph with {graph.number_of_nodes} nodes into {num_clusters} clusters...")

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
        for node, cluster_id in partition.items(): clusters[cluster_id].append(node)
        cluster_list = list(clusters.values())
        print(f"  Graph partitioned into {len(cluster_list)} clusters.")
        return cluster_list

    def _create_subgraph_data_for_model(self, model_type: str, full_graph_obj: DirectedNgramGraph,
                                        full_features: torch.Tensor, full_labels: torch.Tensor,
                                        node_subset: torch.Tensor) -> Data:
        """
        Creates a valid, self-contained PyG Data object for a subgraph of nodes.
        This is the critical fix for clustered training. It re-indexes edges.
        """
        sub_x = full_features[node_subset]
        sub_y = full_labels[node_subset]

        data_dict = {'x': sub_x, 'y': sub_y, 'original_indices': node_subset}

        # Use torch_geometric.utils.subgraph to get re-indexed edges for the subset of nodes
        if model_type == 'directgcn':
            for name, matrix in [('in', full_graph_obj.mathcal_A_in), ('out', full_graph_obj.mathcal_A_out),
                                 ('undirected_norm', full_graph_obj.A_undirected_norm_sparse)]:
                sub_edge_index, sub_edge_weight = subgraph(
                    subset=node_subset, edge_index=matrix.indices(), edge_attr=matrix.values(),
                    relabel_nodes=True, num_nodes=full_graph_obj.number_of_nodes
                )
                data_dict[f'edge_index_{name}'] = sub_edge_index
                data_dict[f'edge_weight_{name}'] = sub_edge_weight
        elif model_type == 'rgcn':
            sub_edge_index_out, _ = subgraph(node_subset, full_graph_obj.A_out_w.indices(), relabel_nodes=True, num_nodes=full_graph_obj.number_of_nodes)
            sub_edge_index_in, _ = subgraph(node_subset, full_graph_obj.A_in_w.indices(), relabel_nodes=True, num_nodes=full_graph_obj.number_of_nodes)
            data_dict['edge_index'] = torch.cat([sub_edge_index_out, sub_edge_index_in], dim=1)
            data_dict['edge_type'] = torch.cat([
                torch.zeros(sub_edge_index_out.size(1), dtype=torch.long),
                torch.ones(sub_edge_index_in.size(1), dtype=torch.long)
            ])
        elif model_type == 'tongdigcn':
            data_dict['edge_index'], _ = subgraph(node_subset, full_graph_obj.A_out_w.indices(), relabel_nodes=True, num_nodes=full_graph_obj.number_of_nodes)
            data_dict['edge_index_backward'], _ = subgraph(node_subset, full_graph_obj.A_in_w.indices(), relabel_nodes=True, num_nodes=full_graph_obj.number_of_nodes)
        else:
            raise ValueError(f"Cannot create subgraph data for unknown model type: {model_type}")

        return Data.from_dict(data_dict)

    def _prepare_data_for_model(self, model_type: str, graph: DirectedNgramGraph, features: torch.Tensor, labels: torch.Tensor) -> Data:
        """Prepares a PyG Data object tailored to the specific model's needs for full-batch training."""
        data_dict = {'x': features, 'y': labels}

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
            edge_type_out = torch.zeros(edge_index_out.size(1), dtype=torch.long)
            edge_type_in = torch.ones(edge_index_in.size(1), dtype=torch.long)
            data_dict['edge_index'] = torch.cat([edge_index_out, edge_index_in], dim=1)
            data_dict['edge_type'] = torch.cat([edge_type_out, edge_type_in], dim=0)
        elif model_type == 'tongdigcn':
            data_dict['edge_index'] = graph.A_out_w.indices()
            data_dict['edge_index_backward'] = graph.A_in_w.indices()
        else:
            raise ValueError(f"Cannot prepare data for unknown model type: {model_type}")
        return Data.from_dict(data_dict)

    def _load_id_map(self) -> Optional[Mapping]:
        """Loads the UniProt ID mapping file if configured."""
        DataUtils.print_header("Step 1: Loading Protein ID Mapping (if configured)")
        if getattr(self.config, 'ID_MAPPING_MODE', 'none') != 'none':
            id_mapper_instance = IDMapGenerator(config=self.config)
            id_map_result = id_mapper_instance.generate_id_maps()
            print(f"  ID mapping result of type '{type(id_map_result)}' loaded.")
            return id_map_result
        return {}
