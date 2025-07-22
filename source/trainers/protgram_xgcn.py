# ==============================================================================
# MODULE: trainers/protgram_xgcn.py
# PURPOSE: Unified trainer for GNNs on ProtGram n-gram graphs.
# VERSION: 4.0 (Refactored for maintainability; helpers and labelgen moved)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import math
import random
from contextlib import nullcontext
from typing import Dict, Optional, List, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from tqdm.auto import tqdm
import collections
from configuration.config import Config
from source.data_builders.graph import DirectedNgramGraph
from source.models.gnn.directgcn import DirectGCN
from source.models.gnn.rgcn import RGCN
from source.models.gnn.tongidigcn import TongDiGCN
from source.utils.post import PostUtils
from source.utils.data import DataUtils, IDMapGenerator
from source.utils.models import EmbeddingProcessor
from source.data_builders.xgcn import XGCNDataset


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
        self.label_generator = XGCNDataset(config)
        self.helpers = PostUtils(config)
        print(f"Using device: {self.device}")

    def run(self) -> Dict[str, str]:
        """
        Main execution function. Loops through n-gram levels, trains models,
        and generates final protein embeddings.
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
                initial_features = torch.randn((graph_obj.number_of_nodes, self.config.GCN_1GRAM_INIT_DIM), device=self.device)
            else:
                prev_level_embeds = ngram_embeddings_per_level.get(n - 1)
                prev_level_map = level_ngram_to_idx.get(n - 1)
                if prev_level_embeds is None or prev_level_embeds.size == 0 or prev_level_map is None:
                    print(f"  Cannot proceed for n={n}, previous level embeddings not found or empty.")
                    continue
                initial_features = self.helpers.pool_lower_level_embeddings(graph_obj, prev_level_embeds, prev_level_map)
                if initial_features is None: continue
                initial_features = initial_features.to(self.device)

            task_type = self.config.GCN_TASK_TYPES_PER_LEVEL.get(n, self.config.GCN_DEFAULT_TASK_TYPE)
            labels, num_classes_for_task = self.label_generator.generate_task_labels(graph_obj, task_type)

            model = self._build_model(model_type, n, initial_features.shape[1], num_classes_for_task, graph_obj.number_of_nodes)
            if model is None: continue
            model.to(self.device)

            data = self._prepare_data_for_model(model_type, graph_obj, initial_features, labels)
            optimizer = optim.Adam(model.parameters(), lr=self.config.GCN_LR, weight_decay=self.config.GCN_WEIGHT_DECAY if l2_lambda_val <= 0 else 0.0)

            if self.config.GCN_USE_CLUSTER_TRAINING and graph_obj.number_of_nodes > self.config.GCN_CLUSTER_TRAINING_THRESHOLD_NODES:
                subgraphs = self._create_clustered_subgraphs(graph_obj, data)
                self._train_model_clustered(model, subgraphs, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, task_type, l2_lambda_val, total_nodes_in_level_graph=graph_obj.number_of_nodes)
            else:
                self._train_model_full_batch(model, data, optimizer, self.config.GCN_EPOCHS_PER_LEVEL, task_type, l2_lambda_val)

            ngram_embeddings_per_level[n] = EmbeddingProcessor.extract_gcn_node_embeddings(model, data, graph_obj, self.config, self.device, self._create_clustered_subgraphs)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.GCN_LR_SCHEDULER_PATIENCE, factor=self.config.GCN_LR_SCHEDULER_FACTOR) if self.config.GCN_USE_LR_SCHEDULER else None
        early_stopper = EarlyStopper(patience=self.config.GCN_EARLY_STOPPING_PATIENCE, min_delta=self.config.GCN_EARLY_STOPPING_MIN_DELTA) if self.config.GCN_USE_EARLY_STOPPING else None
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        # CRITICAL FIX: Use cross_entropy for models outputting raw logits. This is more robust.
        criterion = F.cross_entropy
        print(f"  Starting full-batch training for up to {epochs} epochs (Task: {task_type}, L2 lambda: {l2_lambda})...")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                output, _ = model(data=data)
                primary_loss = criterion(output, data.y)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.GCN_LR_SCHEDULER_PATIENCE, factor=self.config.GCN_LR_SCHEDULER_FACTOR) if self.config.GCN_USE_LR_SCHEDULER else None
        early_stopper = EarlyStopper(patience=self.config.GCN_EARLY_STOPPING_PATIENCE, min_delta=self.config.GCN_EARLY_STOPPING_MIN_DELTA) if self.config.GCN_USE_EARLY_STOPPING else None
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        # CRITICAL FIX: Use cross_entropy for models outputting raw logits.
        criterion = F.cross_entropy
        print(f"  Starting Cluster-GCN style training for up to {epochs} epochs on {len(subgraphs)} subgraphs (Task: {task_type})...")
        for epoch in range(1, epochs + 1):
            random.shuffle(subgraphs)
            epoch_loss = 0.0
            for batch_data in tqdm(subgraphs, desc=f"  Epoch {epoch}", leave=False, disable=not self.config.DEBUG_VERBOSE):
                batch_data = batch_data.to(self.device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    output, _ = model(data=batch_data)
                    primary_loss_per_node_avg = criterion(output, batch_data.y)
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


    def _train_model_full_batch(self, model: nn.Module, data: Data, optimizer: torch.optim.Optimizer, epochs: int,
                                task_type: str, l2_lambda: float = 0.0):
        """Full-batch training logic."""
        model.train()
        model.to(self.device)
        data = data.to(self.device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.GCN_LR_SCHEDULER_PATIENCE, factor=self.config.GCN_LR_SCHEDULER_FACTOR) if self.config.GCN_USE_LR_SCHEDULER else None
        early_stopper = EarlyStopper(patience=self.config.GCN_EARLY_STOPPING_PATIENCE, min_delta=self.config.GCN_EARLY_STOPPING_MIN_DELTA) if self.config.GCN_USE_EARLY_STOPPING else None
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        # CRITICAL FIX: Use cross_entropy for models outputting raw logits. This is more robust.
        criterion = F.cross_entropy
        print(f"  Starting full-batch training for up to {epochs} epochs (Task: {task_type}, L2 lambda: {l2_lambda})...")
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                output, _ = model(data=data)
                primary_loss = criterion(output, data.y)
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

    def _train_model_clustered(self, model: nn.Module, subgraphs: List[Data], optimizer: torch.optim.Optimizer,
                               epochs: int, task_type: str, l2_lambda: float = 0.0,
                               total_nodes_in_level_graph: int = 1):
        """Clustered training logic."""
        model.train()
        model.to(self.device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.GCN_LR_SCHEDULER_PATIENCE, factor=self.config.GCN_LR_SCHEDULER_FACTOR) if self.config.GCN_USE_LR_SCHEDULER else None
        early_stopper = EarlyStopper(patience=self.config.GCN_EARLY_STOPPING_PATIENCE, min_delta=self.config.GCN_EARLY_STOPPING_MIN_DELTA) if self.config.GCN_USE_EARLY_STOPPING else None
        scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        # CRITICAL FIX: Use cross_entropy for models outputting raw logits.
        criterion = F.cross_entropy
        print(f"  Starting Cluster-GCN style training for up to {epochs} epochs on {len(subgraphs)} subgraphs (Task: {task_type})...")
        for epoch in range(1, epochs + 1):
            random.shuffle(subgraphs)
            epoch_loss = 0.0
            for batch_data in tqdm(subgraphs, desc=f"  Epoch {epoch}", leave=False, disable=not self.config.DEBUG_VERBOSE):
                batch_data = batch_data.to(self.device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    output, _ = model(data=batch_data)
                    primary_loss_per_node_avg = criterion(output, batch_data.y)
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
            if scheduler: scheduler.step(avg_epoch_loss)
            if early_stopper and early_stopper.early_stop(avg_epoch_loss):
                print(f"  Early stopping triggered at epoch {epoch}. Best loss: {early_stopper.best_loss:.4f}")
                break

    def _create_clustered_subgraphs(self, graph: DirectedNgramGraph, full_data: Data) -> List[Data]:
        """Partitions the graph into subgraphs, including all necessary matrices."""
        if graph.number_of_nodes == 0: return []
        num_clusters_calculated = math.ceil(graph.number_of_nodes / self.config.GCN_TARGET_NODES_PER_CLUSTER)
        num_clusters = max(self.config.GCN_MIN_CLUSTERS, num_clusters_calculated)
        num_clusters = min(num_clusters, self.config.GCN_MAX_CLUSTERS, graph.number_of_nodes)
        print(f"  Partitioning graph with {graph.number_of_nodes} nodes into {num_clusters} clusters (target nodes/cluster: {self.config.GCN_TARGET_NODES_PER_CLUSTER})...")

        # This method requires METIS or python-louvain, which are imported at the top of the original file
        from torch_geometric.utils import to_networkx
        import community as community_louvain

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

        subgraphs = []
        for cluster_nodes in tqdm(cluster_list, desc="  Creating subgraphs", leave=False):
            nodes_tensor_cpu = torch.tensor(cluster_nodes, dtype=torch.long, device='cpu')
            sub_x = full_data.x[nodes_tensor_cpu]
            sub_y = full_data.y[nodes_tensor_cpu] if full_data.y.numel() > 0 else torch.empty(0, dtype=torch.long)
            subgraph_data = Data(x=sub_x, y=sub_y, original_indices=nodes_tensor_cpu)

            mathcal_A_in_cpu = graph.mathcal_A_in.cpu()
            mathcal_A_out_cpu = graph.mathcal_A_out.cpu()
            A_undir_cpu = graph.A_undirected_norm_sparse.cpu()
            A_out_w_cpu = graph.A_out_w.cpu()
            A_in_w_cpu = graph.A_in_w.cpu()

            for model_type in self.config.PROTGRAM_MODELS_TO_TRAIN:
                if model_type == 'directgcn':
                    sub_edge_index_in, sub_edge_weight_in = subgraph(nodes_tensor_cpu, mathcal_A_in_cpu.indices(), mathcal_A_in_cpu.values(), relabel_nodes=True, num_nodes=graph.number_of_nodes)
                    sub_edge_index_out, sub_edge_weight_out = subgraph(nodes_tensor_cpu, mathcal_A_out_cpu.indices(), mathcal_A_out_cpu.values(), relabel_nodes=True, num_nodes=graph.number_of_nodes)
                    sub_edge_index_undir, sub_edge_weight_undir = subgraph(nodes_tensor_cpu, A_undir_cpu.indices(), A_undir_cpu.values(), relabel_nodes=True, num_nodes=graph.number_of_nodes)
                    # ... (assign to subgraph_data) ...
                    subgraph_data.edge_index_in, subgraph_data.edge_weight_in = sub_edge_index_in, sub_edge_weight_in
                    subgraph_data.edge_index_out, subgraph_data.edge_weight_out = sub_edge_index_out, sub_edge_weight_out
                    subgraph_data.edge_index_undirected_norm, subgraph_data.edge_weight_undirected_norm = sub_edge_index_undir, sub_edge_weight_undir
                elif model_type == 'tongdigcn':
                    sub_edge_index_fwd, _ = subgraph(nodes_tensor_cpu, A_out_w_cpu.indices(), relabel_nodes=True, num_nodes=graph.number_of_nodes)
                    sub_edge_index_bwd, _ = subgraph(nodes_tensor_cpu, A_in_w_cpu.indices(), relabel_nodes=True, num_nodes=graph.number_of_nodes)
                    subgraph_data.edge_index, subgraph_data.edge_index_backward = sub_edge_index_fwd, sub_edge_index_bwd
            subgraphs.append(subgraph_data)
        return subgraphs

    def _load_id_map(self) -> Optional[Mapping]:
        """Loads the UniProt ID mapping file if configured."""
        DataUtils.print_header("Step 1: Loading Protein ID Mapping (if configured)")
        if getattr(self.config, 'ID_MAPPING_MODE', 'none') != 'none':
            id_mapper_instance = IDMapGenerator(config=self.config)
            id_map_result = id_mapper_instance.generate_id_maps()
            print(f"  ID mapping result of type '{type(id_map_result)}' loaded.")
            return id_map_result
        return {}