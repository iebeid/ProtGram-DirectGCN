# ==============================================================================
# MODULE: trainers/protgram_xgcn.py
# PURPOSE: Unified trainer for GNNs on ProtGram n-gram graphs.
# VERSION: 7.1 (Corrected data flow for DirectGCN and integrated homophily paths)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import collections
import copy
import gc
import math
import random
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, List, Mapping, Tuple, Any

import community as community_louvain
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import homophily
from torch_geometric.utils import to_networkx
from tqdm.auto import tqdm
from functools import partial

from configuration.config import Config
from source.data_structures.graph import DirectedNgramGraph
from source.data_builders.xgcn import XGCNDataBuilder
from source.experiments.ppi_1 import PPIPipeline
from source.models.factory import ModelFactory
from source.utils.data import DataUtils, IDMapGenerator, FastaUtils, ProtgramDaskHelpers
from source.utils.models import EmbeddingProcessor, EarlyStopper
from source.utils.results import EvaluationReporter


class ProtGramXGCNTrainer:
    """
    Orchestrates the training of different GNN models on the pre-built
    ProtGram n-gram graphs and generates final protein-level embeddings.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_generator = XGCNDataBuilder(config)
        # --- NEW: Use the centralized model factory ---
        self.model_factory = ModelFactory(config, context='protgram')
        print(f"Using device: {self.device}")
        self._loaded_graphs: Dict[int, DirectedNgramGraph] = {}
        # --- NEW: Set seeds for reproducibility of model initialization and training ---
        DataUtils.set_seeds(self.config.RANDOM_STATE)

    def run(self) -> Dict[str, str]:
        """
        Main execution function. Loops through n-gram levels, trains models,
        and generates final protein-level embeddings.
        """
        DataUtils.print_header("PIPELINE STEP: Training ProtGram Models & Generating Embeddings")

        final_protein_embeddings_per_model = {}
        all_attention_data_per_model: Dict[str, Dict[str, Any]] = {}
        id_map = self._load_id_map()

        context = id_map if isinstance(id_map, IDMapGenerator) else nullcontext(id_map)

        with context as mapper:
            for model_type in self.config.PROTGRAM_MODELS_TO_TRAIN:
                DataUtils.print_header(f"Processing Model Type: {model_type.upper()}")
                ngram_embeddings_per_level, hierarchical_attention = self._train_gnns_hierarchically(model_type)

                final_protein_embeddings, protein_pooling_attention = self._pool_to_protein_level(
                    ngram_embeddings_per_level,
                    level_ngram_to_idx=self._get_level_ngram_maps()
                )

                if mapper and final_protein_embeddings:
                    print("  Applying ID mapping to final protein embeddings...")
                    final_protein_embeddings = {mapper.get(k, k): v for k, v in final_protein_embeddings.items()}

                final_protein_embeddings_per_model[model_type] = final_protein_embeddings
                all_attention_data_per_model[model_type] = {
                    "hierarchical": hierarchical_attention,
                    "protein_pooling": protein_pooling_attention
                }

        output_paths = self._save_final_embeddings(final_protein_embeddings_per_model)

        # --- FIX: Only save/visualize attention if enabled in the config ---
        if self.config.PROTGRAM_LOG_ATTENTION_WEIGHTS:
            self._save_and_visualize_attention(all_attention_data_per_model)

        if self.config.PROTGRAM_RUN_SANITY_CHECK_PPI:
            main_model_name_raw = self.config.PROTGRAM_MODELS_TO_TRAIN[0]
            main_model_key = f"ProtGram{main_model_name_raw.capitalize()}"
            # --- FIX: Prioritize the original, non-PCA'd file for the sanity check ---
            # The evaluation pipeline will handle its own PCA, so we avoid double-processing.
            embedding_path_for_check = output_paths.get(main_model_key, output_paths.get(f"{main_model_key}_pca"))
            if embedding_path_for_check:
                self._run_sanity_check_ppi(embedding_path_for_check)

        DataUtils.print_header("ProtGram Embedding PIPELINE STEP FINISHED")
        return output_paths

    def _load_graph_for_level(self, n: int) -> Optional[DirectedNgramGraph]:
        """Loads the graph object for a specific n-gram level, using a cache."""
        if n in self._loaded_graphs:
            return self._loaded_graphs[n]

        graph_obj_path = self.config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{n}.pkl"
        if not graph_obj_path.exists():
            print(f"  Graph object not found for n={n}. Skipping.")
            return None

        graph_obj: DirectedNgramGraph = DataUtils.load_object(str(graph_obj_path))
        if graph_obj is None or graph_obj.number_of_nodes == 0:
            print(f"  Failed to load graph object or graph is empty for n={n}. Skipping.")
            return None

        self._loaded_graphs[n] = graph_obj
        print(f"  Graph for n={n} loaded. Nodes: {graph_obj.number_of_nodes}")
        # Ensure matrices are on CPU for potential multiprocessing in label generation
        graph_obj.A_out_w = graph_obj.A_out_w.cpu()
        graph_obj.A_in_w = graph_obj.A_in_w.cpu()
        graph_obj.A_undirected_norm_sparse = graph_obj.A_undirected_norm_sparse.cpu()
        return graph_obj

    def _get_initial_features_for_level(self, n: int, graph_obj: DirectedNgramGraph,
                                        prev_level_embeddings: Optional[np.ndarray],
                                        prev_level_map: Optional[Dict[str, int]]) -> Optional[Tuple[torch.Tensor, Dict]]:
        """Generates the initial node features for the current n-gram level."""
        if n == 1:
            features = torch.randn((graph_obj.number_of_nodes, self.config.PROTGRAM_1GRAM_INIT_DIM))
            return features, {}
        else:
            if prev_level_embeddings is None or prev_level_embeddings.size == 0 or prev_level_map is None:
                print(f"  Cannot proceed for n={n}, previous level embeddings not found or empty.")
                return None
            result = EmbeddingProcessor.pool_lower_level_embeddings_for_init(
                graph_obj, prev_level_embeddings, prev_level_map,
                strategy=self.config.PROTGRAM_HIERARCHICAL_POOLING_STRATEGY)
            if result is None:
                return None
            features, attention_log = result
            return features, attention_log

    def _train_gnns_hierarchically(self, model_type: str) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        """The main hierarchical training loop."""
        ngram_embeddings_per_level: Dict[int, np.ndarray] = {}
        level_ngram_to_idx: Dict[int, Dict[str, int]] = {}
        hierarchical_attention_per_level: Dict[int, Dict] = {}

        for n in range(1, self.config.PROTGRAM_NGRAM_MAX_N + 1):
            DataUtils.print_header(f"Processing N-gram Level: n = {n} for model '{model_type}'")

            graph_obj = self._load_graph_for_level(n)
            if not graph_obj: continue

            level_ngram_to_idx[n] = graph_obj.get_node_to_idx_map()
            prev_embeds = ngram_embeddings_per_level.get(n - 1)
            prev_map = level_ngram_to_idx.get(n - 1)

            feature_result = self._get_initial_features_for_level(n, graph_obj, prev_embeds, prev_map)
            if feature_result is None: continue
            initial_features, hierarchical_attention = feature_result

            if hierarchical_attention and self.config.PROTGRAM_LOG_ATTENTION_WEIGHTS:
                hierarchical_attention_per_level[n] = hierarchical_attention

            task_type = self.config.PROTGRAM_TASK_TYPES_PER_LEVEL.get(n, self.config.PROTGRAM_DEFAULT_TASK_TYPE)
            labels, num_classes_for_task = self.label_generator.generate_task_labels(graph_obj, task_type)

            A_homo_norm, A_hetero_norm = None, None
            use_homo_hetero_paths_for_level = False
            if model_type == 'directgcn' and labels is not None:
                homophily_ratio = homophily(graph_obj.A_undirected_norm_sparse.indices(), labels,
                                            method='edge')
                is_heterophilic = homophily_ratio < self.config.GCN_HETEROPHILY_THRESHOLD
                print(f"  Graph n={n} Homophily Ratio: {homophily_ratio:.4f}. Is Heterophilic? -> {is_heterophilic}")
                if is_heterophilic:
                    print(f"  -> Enabling specialized homophily/heterophily paths for DirectGCN at n={n}.")
                    use_homo_hetero_paths_for_level = True
                    split_result = graph_obj.split_edges_by_homophily(labels)
                    if split_result:
                        A_homo_norm, A_hetero_norm = split_result

            model = self.model_factory.create_model(
                model_name=model_type, in_channels=initial_features.shape[1],
                num_classes=num_classes_for_task, graph_obj=graph_obj,
                use_homo_hetero_paths=use_homo_hetero_paths_for_level, n_val=n
            )
            if model is None: continue

            data = Data(x=initial_features, y=labels, graph_obj=graph_obj,
                        A_homo_norm=A_homo_norm, A_hetero_norm=A_hetero_norm)
            optimizer = optim.Adam(model.parameters(), lr=self.config.PROTGRAM_LR, weight_decay=self.config.PROTGRAM_WEIGHT_DECAY)

            self._train_single_level(model, graph_obj, data, optimizer, use_homo_hetero_paths_for_level)

            prepare_func = partial(ProtgramDaskHelpers.prepare_pyg_data_from_protgram_graph,
                                   use_homo_hetero_paths=use_homo_hetero_paths_for_level,
                                   A_homo_norm=A_homo_norm, A_hetero_norm=A_hetero_norm)
            ngram_embeddings_per_level[n] = EmbeddingProcessor.extract_gcn_node_embeddings(
                model, data, graph_obj, self.config, self.device,
                prepare_data_func=prepare_func,
                create_clustered_subgraphs_func=lambda g: self._partition_graph(g)
            )

            print(f"  Generated {ngram_embeddings_per_level[n].shape[0]} embeddings of dim {ngram_embeddings_per_level[n].shape[1]} for n={n}.")
            del model, data, graph_obj, initial_features, labels, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return ngram_embeddings_per_level, hierarchical_attention_per_level

    def _train_single_level(self, model: nn.Module, graph_obj: DirectedNgramGraph, data: Data, optimizer: torch.optim.Optimizer, use_homo_hetero_paths: bool):
        """Orchestrates the training for a single level, choosing between full-batch and clustered training."""
        task_type = self.config.PROTGRAM_TASK_TYPES_PER_LEVEL.get(graph_obj.n_value, self.config.PROTGRAM_DEFAULT_TASK_TYPE)

        if self.config.PROTGRAM_USE_CLUSTER_TRAINING and graph_obj.number_of_nodes > self.config.PROTGRAM_CLUSTER_TRAINING_THRESHOLD_NODES:
            node_partitions = self._partition_graph(graph_obj)
            self._train_single_level_clustered(model, data, node_partitions, optimizer, self.config.PROTGRAM_EPOCHS_PER_LEVEL, task_type)
        else:
            self._train_single_level_full_batch(model, data, optimizer, self.config.PROTGRAM_EPOCHS_PER_LEVEL, task_type, use_homo_hetero_paths)

    def _train_single_level_full_batch(self, model: nn.Module, data: Data, optimizer: torch.optim.Optimizer, epochs: int,
                                       task_type: str, use_homo_hetero_paths: bool):
        """Full-batch training logic for a single GNN level."""
        model.train()
        model.to(self.device)
        # --- FIX: Use the centralized data preparation utility --- # noqa
        full_data_gpu = ProtgramDaskHelpers.prepare_pyg_data_from_protgram_graph(
            model_type=model.__class__.__name__.lower(), graph=data.graph_obj,
            features=data.x, labels=data.y, use_homo_hetero_paths=use_homo_hetero_paths,
            A_homo_norm=getattr(data, 'A_homo_norm', None),
            A_hetero_norm=getattr(data, 'A_hetero_norm', None)
        ).to(self.device)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.PROTGRAM_LR_SCHEDULER_PATIENCE, factor=self.config.PROTGRAM_LR_SCHEDULER_FACTOR) if self.config.PROTGRAM_USE_LR_SCHEDULER else None
        early_stopper = EarlyStopper(patience=self.config.PROTGRAM_EARLY_STOPPING_PATIENCE, min_delta=self.config.PROTGRAM_EARLY_STOPPING_MIN_DELTA) if self.config.PROTGRAM_USE_EARLY_STOPPING else None
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))

        criterion = F.cross_entropy
        print(f"  Starting full-batch training for up to {epochs} epochs (Task: {task_type})...")
        for epoch in range(1, epochs + 1):
            # --- CONCEPTUAL CHANGE FOR MASKED NODE PREDICTION ---
            # If the task is 'masked_node', we need to generate a new mask for each epoch.
            if task_type == 'masked_node':
                masked_features, masked_indices, original_labels = self.label_generator.generate_masked_node_task(                    graph_obj=full_data_gpu.graph_obj, features=data.x,
                    masking_fraction=self.config.PROTGRAM_MASKED_NODE_FRACTION
                )
                # Update the data object for this epoch's forward pass
                epoch_data = full_data_gpu.clone()
                epoch_data.x = masked_features.to(self.device)
                masked_indices = masked_indices.to(self.device)
                original_labels = original_labels.to(self.device)
            else:
                epoch_data = full_data_gpu

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                output, _ = model(data=epoch_data)
                if task_type == 'masked_node':
                    loss = criterion(output[masked_indices], original_labels)
                else: # Original 'next_node' or 'community' logic
                    loss = criterion(output, epoch_data.y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler: scheduler.step(loss)
            if self.config.DEBUG_VERBOSE and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                print(f"    Epoch: {epoch:03d}, Loss: {loss.item():.4f}")
            if early_stopper and early_stopper.early_stop(loss.item()):
                print(f"  Early stopping triggered at epoch {epoch}. Best loss: {early_stopper.best_loss:.4f}")
                break

    # --- FIX: Add the 'use_homo_hetero_paths' parameter to prevent a TypeError ---
    def _train_single_level_clustered(self, model: nn.Module, full_data: Data, node_partitions: List[List[int]], optimizer: torch.optim.Optimizer, epochs: int, task_type: str):
        """Clustered training logic for a single GNN level."""
        model.train()
        model.to(self.device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.PROTGRAM_LR_SCHEDULER_PATIENCE, factor=self.config.PROTGRAM_LR_SCHEDULER_FACTOR) if self.config.PROTGRAM_USE_LR_SCHEDULER else None
        early_stopper = EarlyStopper(patience=self.config.PROTGRAM_EARLY_STOPPING_PATIENCE, min_delta=self.config.PROTGRAM_EARLY_STOPPING_MIN_DELTA) if self.config.PROTGRAM_USE_EARLY_STOPPING else None
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))
        criterion = F.cross_entropy
        print(f"  Starting Cluster-GCN style training for up to {epochs} epochs on {len(node_partitions)} subgraphs (Task: {task_type})...")

        for epoch in range(1, epochs + 1):
            random.shuffle(node_partitions)
            epoch_loss = 0.0
            for node_idx_batch in tqdm(node_partitions, desc=f"  Epoch {epoch}", leave=False, disable=not self.config.DEBUG_VERBOSE):
                # --- DEFINITIVE FIX: Pass the homophily/heterophily matrices to the subgraph creator ---
                subgraph_data = full_data.graph_obj.create_subgraph_data_for_model(
                    model_type=model.__class__.__name__.lower(),
                    full_features=full_data.x,
                    full_labels=full_data.y,
                    node_subset=torch.tensor(node_idx_batch, dtype=torch.long),
                    # Pass the pre-calculated matrices from the full data object
                    A_homo_norm=getattr(full_data, 'A_homo_norm', None),
                    A_hetero_norm=getattr(full_data, 'A_hetero_norm', None)
                ).to(self.device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
                    # --- DEFINITIVE FIX: Implement masked_node logic for clustered training ---
                    if task_type == 'masked_node':
                        num_subgraph_nodes = subgraph_data.num_nodes
                        num_to_mask = int(num_subgraph_nodes * self.config.PROTGRAM_MASKED_NODE_FRACTION)

                        if num_to_mask > 0:
                            # Indices are relative to the subgraph for this batch
                            permuted_subgraph_indices = torch.randperm(num_subgraph_nodes, device=self.device)
                            subgraph_masked_indices = permuted_subgraph_indices[:num_to_mask]

                            # The ground truth labels are the original, full-graph node IDs
                            original_node_labels = subgraph_data.original_indices[subgraph_masked_indices]

                            # Create a masked version of the subgraph features for this batch
                            masked_subgraph_features = subgraph_data.x.clone()
                            masked_subgraph_features[subgraph_masked_indices] = 0.0
                            subgraph_data.x = masked_subgraph_features

                            output, _ = model(data=subgraph_data)
                            loss = criterion(output[subgraph_masked_indices], original_node_labels)
                        else:
                            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    else:  # Original logic for community/next_node
                        output, _ = model(data=subgraph_data)
                        loss = criterion(output, subgraph_data.y)
                scaler.scale(loss).backward()
                # --- FIX: Correctly order gradient clipping and scaling ---
                # Unscale gradients before clipping to ensure we clip the true gradients, not the scaled ones.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        num_clusters_calculated = math.ceil(graph.number_of_nodes / self.config.PROTGRAM_TARGET_NODES_PER_CLUSTER)
        num_clusters = max(self.config.PROTGRAM_MIN_CLUSTERS, num_clusters_calculated)
        num_clusters = min(num_clusters, self.config.PROTGRAM_MAX_CLUSTERS, graph.number_of_nodes)
        print(f"  Partitioning graph with {graph.number_of_nodes} nodes into {num_clusters} clusters...")

        A_combined_cpu = (graph.A_in_w.cpu() + graph.A_out_w.cpu()).coalesce()
        g_nx = to_networkx(Data(edge_index=A_combined_cpu.indices(), edge_attr=A_combined_cpu.values(), num_nodes=graph.number_of_nodes), to_undirected=True, edge_attrs=['edge_attr']) # type: ignore

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

    def _load_id_map(self) -> Optional[Mapping]:
        """Loads the UniProt ID mapping file if configured."""
        DataUtils.print_header("Step 1: Loading Protein ID Mapping (if configured)")
        if getattr(self.config, 'ID_MAPPING_MODE', 'none') != 'none':
            id_mapper_instance = IDMapGenerator(config=self.config)
            id_map_result = id_mapper_instance.generate_id_maps()
            print(f"  ID mapping result of type '{type(id_map_result)}' loaded.")
            return id_map_result
        return None

    def _get_level_ngram_maps(self) -> Dict[int, Dict[str, int]]:
        """Returns the node_to_idx maps for all loaded graph levels."""
        return {n: graph.get_node_to_idx_map() for n, graph in self._loaded_graphs.items()}

    def _pool_to_protein_level(self, ngram_embeddings_per_level: Dict[int, np.ndarray],
                               level_ngram_to_idx: Dict[int, Dict[str, int]]) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[int, float]]]:
        """Pools the final n-gram embeddings to the protein level."""
        final_n = self.config.PROTGRAM_NGRAM_MAX_N
        final_level_embeddings = ngram_embeddings_per_level.get(final_n)
        final_level_map = level_ngram_to_idx.get(final_n)

        if final_level_embeddings is None or final_level_map is None:
            print("  Final level embeddings or map not found. Cannot perform protein-level pooling.")
            return {}, {}

        protein_sequences = list(FastaUtils.parse_sequences(self.config.SEQUENCE_FILE_PATHS))

        return EmbeddingProcessor.pool_ngram_embeddings_for_protein_fast(
            protein_sequences=protein_sequences,
            n_val=final_n,
            ngram_map=final_level_map,
            ngram_embeddings=final_level_embeddings,
            strategy=self.config.PROTGRAM_PROTEIN_POOLING_STRATEGY
        )



    def _run_sanity_check_ppi(self, embedding_path: str):
        """Runs a quick, small-scale PPI evaluation as a sanity check."""
        DataUtils.print_header("Running Sanity Check PPI Evaluation")
        print(f"  Using embeddings from: {Path(embedding_path).name}")
        sanity_config = copy.deepcopy(self.config)

        # Override config for a quick run
        sanity_config.EVAL_EPOCHS = self.config.PROTGRAM_SANITY_CHECK_EPOCHS
        sanity_config.EVAL_N_FOLDS = 2  # A minimal number of folds for a quick check
        sanity_config.EVAL_GENERATE_SHAP_SUMMARY = False  # Disable for speed
        sanity_config.PLOT_TRAINING_HISTORY = False  # Disable for speed

        # Set the specific embedding file to evaluate
        model_name_for_eval = Path(embedding_path).stem.replace('_pca', '').replace(str(self.config.PCA_TARGET_DIMENSION), '')
        sanity_config.LP_EMBEDDING_FILES_TO_EVALUATE = [
            {"name": model_name_for_eval, "path": embedding_path}
        ]

        try:
            # Instantiate and run the pipeline with the modified config
            ppi_evaluator = PPIPipeline(sanity_config)
            ppi_evaluator.run(use_dummy_data=False)
        except Exception as e:
            print(f"  ❌ Sanity check PPI evaluation failed with an error: {e}")
            traceback.print_exc()

        print("--- Sanity Check PPI Evaluation Finished ---")