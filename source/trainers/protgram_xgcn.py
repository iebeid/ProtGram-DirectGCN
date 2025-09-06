# ==============================================================================
# MODULE: trainers/protgram_xgcn.py
# PURPOSE: Unified trainer for GNNs on ProtGram n-gram graphs.
# VERSION: 8.0 (Refactored to use centralized IDMapper singleton)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import math
import copy
import gc
import random
import traceback
from pathlib import Path
from typing import Dict, Optional, List, Mapping, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import homophily
from tqdm.auto import tqdm
from functools import partial
import mlflow
from configuration.config import Config
from source.data_structures.direct_ngram_graph import DirectedNgramGraph
from source.data_builders.xgcn import XGCNDataBuilder
from source.utils.fs.file_utils import FileUtils
from source.experiments.ppi_1 import PPIPipeline
from source.models.factory import ModelFactory
from source.utils.data.data_utils import DataUtils
from source.utils.data.id_mapper import IDMapper
from source.utils.data.fasta_utils import FastaUtils
from source.utils.data.ground_truth_loader import GroundTruthLoader
from source.utils.post.embedding_loader import EmbeddingLoader
from source.utils.data.protgram_helper import ProtgramDaskHelpers
from source.utils.post.embedding_processor import EmbeddingProcessor
from source.utils.models.early_stopper import EarlyStopper
from source.utils.results.evaluation_reporter import EvaluationReporter


class ProtGramXGCNTrainer:
    """
    Orchestrates the hierarchical training of GNN models on the pre-built
    ProtGram n-gram graphs.

    The core workflow is as follows:
    1. For each n-gram level (from n=1 to n_max):
       a. Load the corresponding n-gram graph.
       b. Initialize its node features. For n=1, this is random. For n>1,
          features are pooled from the embeddings of the (n-1) level graph.
       c. Train a GNN on this graph to produce node (n-gram) embeddings.
    2. After all levels are trained, pool the n-gram embeddings for each protein
       sequence to generate a final, fixed-size vector for each protein.
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
        and generates final protein-level embeddings with raw IDs.
        """
        mlflow.set_experiment(self.config.MLFLOW_PROTGRAM_XGCN_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"ProtGram-XGCN_{Path(self.config.SEQUENCE_FILE_PATHS[0]).stem}"):
            DataUtils.print_header("PIPELINE STEP: Training ProtGram Models & Generating Embeddings")

            final_protein_embeddings_per_model: Dict[str, Dict[str, np.ndarray]] = {}
            final_attention_logs: Dict[str, Dict] = {}  # Initialize the aggregator
            protein_sequences = list(FastaUtils.parse_sequences(self.config.SEQUENCE_FILE_PATHS))

            for model_type in self.config.PROTGRAM_MODELS_TO_TRAIN:
                if model_type == 'directgcn':
                    for gating_mode in ['vector', None]:
                        gating_mode_str = gating_mode if gating_mode is not None else 'none'
                        DataUtils.print_header(f"Processing Model Type: {model_type.upper()} (Gating: {gating_mode_str})")

                        # Correctly capture the attention logs from the hierarchical trainer
                        ngram_embeddings_per_level, attention_logs_for_model = self._train_gnns_hierarchically(
                            model_type, gating_coeff_mode=gating_mode
                        )
                        final_attention_logs.update(attention_logs_for_model)  # Aggregate logs

                        pooled_results = self._pool_to_protein_level(
                            ngram_embeddings_per_level, protein_sequences,
                            level_ngram_to_idx=self._get_level_ngram_maps()
                        )
                        for n, (pooled_embeddings, _) in pooled_results.items():
                            final_protein_embeddings_per_model[f"{model_type}_{gating_mode_str}_gating_n{n}"] = pooled_embeddings
                else:
                    DataUtils.print_header(f"Processing Model Type: {model_type.upper()}")

                    # Correctly capture the attention logs from the hierarchical trainer
                    ngram_embeddings_per_level, attention_logs_for_model = self._train_gnns_hierarchically(model_type)
                    final_attention_logs.update(attention_logs_for_model)  # Aggregate logs

                    pooled_results = self._pool_to_protein_level(
                        ngram_embeddings_per_level, protein_sequences,
                        level_ngram_to_idx=self._get_level_ngram_maps()
                    )
                    for n, (pooled_embeddings, _) in pooled_results.items():
                        final_protein_embeddings_per_model[f"{model_type}_n{n}"] = pooled_embeddings

            # This function now saves the raw (unmapped) embeddings and returns their paths.
            output_paths = self._save_final_embeddings(final_protein_embeddings_per_model)

            # This method is now responsible for generating embeddings and attention logs.
            if self.config.PROTGRAM_LOG_ATTENTION_WEIGHTS:
                for model_name, attention_log in final_attention_logs.items():
                    if attention_log:
                        log_path = self.config.RESULTS_GCN_EMBEDDINGS_DIR / f"attention_log_{model_name}.json"
                        FileUtils.save_json(attention_log, log_path)
                        print(f"  Saved attention log for '{model_name}' to {log_path.name}")

            DataUtils.print_header("ProtGram Embedding PIPELINE STEP FINISHED")
            return output_paths

    def _pool_to_protein_level(self, ngram_embeddings_per_level: Dict[int, np.ndarray],
                               protein_sequences: List[Tuple[str, str]],
                               level_ngram_to_idx: Dict[int, Dict[str, int]]) -> Dict[int, Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]]:
        
        pooled_results = {}
        for n in range(1, self.config.PROTGRAM_NGRAM_MAX_N + 1):
            DataUtils.print_header(f"Step 3: Pooling N-Gram Embeddings to Protein Level for n={n}")
            level_embeddings = ngram_embeddings_per_level.get(n)
            level_map = level_ngram_to_idx.get(n)

            if level_embeddings is None or level_map is None:
                print(f"  ERROR: N-gram embeddings for n={n} are not available. Cannot perform protein-level pooling.")
                continue

            # Use the highly optimized pooling function from the processor
            pooled_embeddings, attention_log = EmbeddingProcessor.pool_ngram_embeddings_for_protein_fast(
                protein_sequences=protein_sequences,
                n_val=n,
                ngram_map=level_map,
                ngram_embeddings=level_embeddings,
                strategy=self.config.PROTGRAM_PROTEIN_POOLING_STRATEGY
            )
            pooled_results[n] = (pooled_embeddings, attention_log)
        return pooled_results

    def _save_final_embeddings(self, embeddings_per_model: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, str]:
        """Saves final protein embeddings and their PCA versions to H5 files."""
        output_paths = {}
        for model_name, embeddings in embeddings_per_model.items():
            if not embeddings:
                print(f"  No embeddings generated for model '{model_name}'. Skipping save.")
                continue

            output_dir = self.config.RESULTS_GCN_EMBEDDINGS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get embedding dimension for filename
            first_emb = next(iter(embeddings.values()), None)
            if first_emb is None: continue
            dim = first_emb.shape[0]

            output_path = output_dir / f"{model_name}_dim{dim}.h5"

            FileUtils.write_h5(embeddings, output_path, f"Writing H5 for {model_name}")
            output_paths[model_name] = str(output_path)

            if self.config.PCA_TARGET_DIMENSION > 0:
                pca_path = EmbeddingProcessor.apply_pca_to_h5(
                    input_h5_path=output_path,
                    output_dir=output_dir,
                    target_dimension=self.config.PCA_TARGET_DIMENSION,
                    random_seed=self.config.RANDOM_STATE
                )
                if str(pca_path) != str(output_path):
                    output_paths[f"{model_name}_pca"] = str(pca_path)
        return output_paths


    def _train_gnns_hierarchically(self, model_type: str, gating_coeff_mode: Optional[str] = "vector") -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
        """
        The main hierarchical training loop. It iterates from n=1 to n_max,
        training a GNN at each level and using its output embeddings to initialize
        the features for the next level.

        Returns:
            A tuple of (ngram_embeddings_per_level, hierarchical_attention_per_level).
        """
        ngram_embeddings_per_level: Dict[int, np.ndarray] = {}
        level_ngram_to_idx: Dict[int, Dict[str, int]] = {}
        final_attention_logs: Dict[str, Dict] = {}
        hierarchical_attention_per_level: Dict[int, Dict] = {}

        for n in range(1, self.config.PROTGRAM_NGRAM_MAX_N + 1):
            DataUtils.print_header(f"Processing N-gram Level: n = {n} for model '{model_type}'")

            graph_obj = self._load_graph_for_level(n)
            if not graph_obj: continue

            # Store the node-to-ID map for this level
            level_ngram_to_idx[n] = graph_obj.get_node_to_idx_map()
            prev_embeds = ngram_embeddings_per_level.get(n - 1)
            prev_map = level_ngram_to_idx.get(n - 1)

            # Get initial features (random for n=1, pooled from n-1 for n>1)
            feature_result = self._get_initial_features_for_level(n, graph_obj, prev_embeds, prev_map)
            if feature_result is None: continue
            initial_features, hierarchical_attention = feature_result

            bn = torch.nn.BatchNorm1d(initial_features.shape[1]).to(self.device)
            # --- FIX: Detach the features from the computation graph after normalization ---
            # This prevents the "trying to backward through the graph a second time" error in the training loop.
            initial_features = bn(initial_features.to(self.device)).detach().cpu()

            if hierarchical_attention and self.config.PROTGRAM_LOG_ATTENTION_WEIGHTS:
                hierarchical_attention_per_level[n] = hierarchical_attention

            # Determine the training task for this level (e.g., community detection, masked node prediction)
            task_type = self.config.PROTGRAM_TASK_TYPES_PER_LEVEL.get(n, self.config.PROTGRAM_DEFAULT_TASK_TYPE)
            labels, num_classes_for_task = self.label_generator.generate_task_labels(graph_obj, task_type)

            # Special handling for DirectGCN on heterophilic graphs
            A_homo_norm, A_hetero_norm = None, None
            use_homo_hetero_paths_for_level = False
            if model_type == 'directgcn' and labels is not None:
                # Use a robust homophily computation on the undirected (no self-loops) structure
                edge_index_undir = graph_obj.A_undirected_w.indices()
                homophily_ratio = ProtgramDaskHelpers.safe_homophily(edge_index_undir, labels, default=0.5)
                is_heterophilic = homophily_ratio < self.config.GCN_HETEROPHILY_THRESHOLD
                print(f"  Graph n={n} Homophily Ratio: {homophily_ratio:.4f}. Is Heterophilic? -> {is_heterophilic}")
                if is_heterophilic:
                    print(f"  -> Enabling specialized homophily/heterophily paths for DirectGCN at n={n}.")
                    use_homo_hetero_paths_for_level = True
                    split_result = DirectedNgramGraph.split_edges_by_homophily(
                        graph_obj.A_out_w, graph_obj.number_of_nodes, labels
                    )
                    if split_result: A_homo_norm, A_hetero_norm = split_result

            model = self.model_factory.create_model(
                model_name=model_type, in_channels=initial_features.shape[1],
                num_classes=num_classes_for_task, graph_obj=graph_obj,
                use_homo_hetero_paths=use_homo_hetero_paths_for_level, n_val=n,
                gating_mode=gating_coeff_mode
            )
            if model is None: continue

            data = Data(x=initial_features, y=labels, graph_obj=graph_obj,
                        A_homo_norm=A_homo_norm, A_hetero_norm=A_hetero_norm)
            optimizer = optim.Adam(model.parameters(), lr=self.config.PROTGRAM_LR, weight_decay=self.config.PROTGRAM_WEIGHT_DECAY)

            # Train the model for this level
            self._train_single_level(model, graph_obj, data, optimizer)

            prepare_func = partial(ProtgramDaskHelpers.prepare_pyg_data_from_protgram_graph,
                                   A_homo_norm=A_homo_norm, A_hetero_norm=A_hetero_norm)
            ngram_embeddings_per_level[n] = EmbeddingProcessor.extract_gcn_node_embeddings(
                model, data, graph_obj, self.config, self.device,
                prepare_data_func=prepare_func,
                create_clustered_subgraphs_func=lambda g: self._partition_graph(g)
            )

            # --- NEW: Capture the attention log for the final level ---
            if n == self.config.PROTGRAM_NGRAM_MAX_N:
                model_run_name = f"{model_type}_{gating_coeff_mode}_gating_n{n}" if model_type == 'directgcn' else f"{model_type}_n{n}"
                final_attention_logs[model_run_name] = hierarchical_attention_per_level.get(n, {})

            del model, data, graph_obj, initial_features, labels, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return ngram_embeddings_per_level, hierarchical_attention_per_level

    def _train_single_level(self, model: nn.Module, graph_obj: DirectedNgramGraph, data: Data, optimizer: torch.optim.Optimizer):
        """Orchestrates the training for a single level, choosing between full-batch and clustered training."""
        # Determine the task type for this level (e.g., community detection, masked node prediction)
        task_type = self.config.PROTGRAM_TASK_TYPES_PER_LEVEL.get(graph_obj.n_value, self.config.PROTGRAM_DEFAULT_TASK_TYPE)

        # If the graph is very large, use clustered training to avoid OOM errors.
        # Otherwise, use standard full-batch training.
        if self.config.PROTGRAM_USE_CLUSTER_TRAINING and graph_obj.number_of_nodes > self.config.PROTGRAM_CLUSTER_TRAINING_THRESHOLD_NODES:
            node_partitions = self._partition_graph(graph_obj)
            self._train_single_level_clustered(model, data, node_partitions, optimizer, self.config.PROTGRAM_EPOCHS_PER_LEVEL, task_type)
        else:
            self._train_single_level_full_batch(model, data, optimizer, self.config.PROTGRAM_EPOCHS_PER_LEVEL, task_type)

    def _train_single_level_full_batch(self, model: nn.Module, data: Data, optimizer: torch.optim.Optimizer, epochs: int,
                                       task_type: str):
        """Full-batch training logic for a single GNN level."""
        model.train()
        model.to(self.device)
        # --- FIX: Use the centralized data preparation utility --- # noqa
        full_data_gpu = ProtgramDaskHelpers.prepare_pyg_data_from_protgram_graph(
            model_type=model.__class__.__name__.lower(), graph=data.graph_obj, features=data.x, labels=data.y,
            A_homo_norm=getattr(data, 'A_homo_norm', None),
            A_hetero_norm=getattr(data, 'A_hetero_norm', None)
        ).to(self.device)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.PROTGRAM_LR_SCHEDULER_PATIENCE, factor=self.config.PROTGRAM_LR_SCHEDULER_FACTOR) if self.config.PROTGRAM_USE_LR_SCHEDULER else None
        early_stopper = EarlyStopper(patience=self.config.PROTGRAM_EARLY_STOPPING_PATIENCE, min_delta=self.config.PROTGRAM_EARLY_STOPPING_MIN_DELTA) if self.config.PROTGRAM_USE_EARLY_STOPPING else None

        # --- DEFINITIVE FIX for NaN Loss: Disable mixed-precision for ALL levels. ---
        # While AMP provides a speedup, it has proven to be numerically unstable
        # for this specific architecture, causing intermittent NaN loss values, especially
        # for n>1 graphs. Forcing float32 provides stability.
        use_amp = False
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # --- NEW: Add gradient accumulation ---
        accumulation_steps = self.config.PROTGRAM_GRADIENT_ACCUMULATION_STEPS
        if accumulation_steps > 1:
            print(f"  Gradient accumulation enabled with {accumulation_steps} steps.")
        optimizer.zero_grad()

        criterion = F.cross_entropy
        print(f"  Starting full-batch training for up to {epochs} epochs (Task: {task_type})...")
        for epoch in range(1, epochs + 1):  # noqa
            # --- CONCEPTUAL CHANGE FOR MASKED NODE PREDICTION ---
            # If the task is 'masked_node', we need to generate a new mask for each epoch.
            if task_type == 'masked_node':
                masked_features, masked_indices, original_labels = self.label_generator.generate_masked_node_task(                    graph_obj=full_data_gpu.graph_obj, features=data.x,
                    # The task is to predict the original node IDs from the masked features
                    masking_fraction=self.config.PROTGRAM_MASKED_NODE_FRACTION
                )
                # Update the data object for this epoch's forward pass
                epoch_data = full_data_gpu.clone()
                epoch_data.x = masked_features.to(self.device)
                masked_indices = masked_indices.to(self.device)
                original_labels = original_labels.to(self.device)
            else:
                epoch_data = full_data_gpu

            with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                output, _ = model(data=epoch_data)
                # Calculate loss based on the specific task for this level
                if task_type == 'masked_node':
                    loss = criterion(output[masked_indices], original_labels)
                else: # Original 'next_node' or 'community' logic
                    loss = criterion(output, epoch_data.y)

            # --- NEW: Calculate training metrics for more detailed logging ---
            train_metrics = {}
            with torch.no_grad():
                if task_type == 'masked_node':
                    preds = output[masked_indices].argmax(dim=-1)
                    correct = (preds == original_labels).sum().item()
                    total = len(original_labels)
                    train_metrics['train_acc'] = correct / total if total > 0 else 0.0
                else:
                    preds = output.argmax(dim=-1)
                    correct = (preds == epoch_data.y).sum().item()
                    total = len(epoch_data.y)
                    train_metrics['train_acc'] = correct / total if total > 0 else 0.0

            # --- Backward pass & Gradient Accumulation ---
            unnormalized_loss = loss.item()
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (epoch % accumulation_steps) == 0 or (epoch == epochs):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if scheduler: scheduler.step(unnormalized_loss)
            # --- NEW: Enhanced logging with more metrics ---
            if self.config.DEBUG_VERBOSE and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                current_lr = optimizer.param_groups[0]['lr']
                log_str = (f"    Epoch: {epoch:03d}, Loss: {unnormalized_loss:.4f}, "
                           f"Train Acc: {train_metrics.get('train_acc', 0.0):.4f}, "
                           f"LR: {current_lr:.6f}")
                print(log_str)
            if early_stopper and early_stopper.early_stop(unnormalized_loss):
                print(f"  Early stopping triggered at epoch {epoch}. Best loss: {early_stopper.best_loss:.4f}")
                break

    def _calculate_loss(self, model: nn.Module, data: Data, criterion, task_type: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass and calculates the loss for a given task.
        This helper centralizes the loss logic for both full-batch and clustered training.

        Returns:
            A tuple of (loss, predictions, ground_truth_labels).
        """
        if task_type == 'masked_node':
            masked_features, masked_indices, original_labels = self.label_generator.generate_masked_node_task(
                graph_obj=data.graph_obj, features=data.x,
                masking_fraction=self.config.PROTGRAM_MASKED_NODE_FRACTION
            )
            # Use a cloned data object for the forward pass to avoid modifying the original
            epoch_data = data.clone()
            epoch_data.x = masked_features.to(self.device)
            masked_indices = masked_indices.to(self.device)
            original_labels = original_labels.to(self.device)

            output, _ = model(data=epoch_data)
            loss = criterion(output[masked_indices], original_labels)
            with torch.no_grad():
                preds = output[masked_indices].argmax(dim=-1)
            return loss, preds, original_labels
        else:  # Handles 'community', 'next_node', etc.
            output, _ = model(data=data)
            loss = criterion(output, data.y)
            with torch.no_grad():
                preds = output.argmax(dim=-1)
            return loss, preds, data.y

    # --- FIX: Add the 'use_homo_hetero_paths' parameter to prevent a TypeError ---
    def _train_single_level_clustered(self, model: nn.Module, full_data: Data, node_partitions: List[List[int]], optimizer: torch.optim.Optimizer, epochs: int, task_type: str):
        """Clustered training logic for a single GNN level."""
        model.train()
        model.to(self.device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.config.PROTGRAM_LR_SCHEDULER_PATIENCE, factor=self.config.PROTGRAM_LR_SCHEDULER_FACTOR) if self.config.PROTGRAM_USE_LR_SCHEDULER else None
        early_stopper = EarlyStopper(patience=self.config.PROTGRAM_EARLY_STOPPING_PATIENCE, min_delta=self.config.PROTGRAM_EARLY_STOPPING_MIN_DELTA) if self.config.PROTGRAM_USE_EARLY_STOPPING else None
        # --- DEFINITIVE FIX for NaN Loss: Disable mixed-precision for ALL levels. ---
        # While AMP provides a speedup, it has proven to be numerically unstable
        # for this specific architecture. Forcing float32 provides stability.
        use_amp = False
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # --- NEW: Add gradient accumulation ---
        accumulation_steps = self.config.PROTGRAM_GRADIENT_ACCUMULATION_STEPS
        if accumulation_steps > 1:
            print(f"  Gradient accumulation enabled with {accumulation_steps} steps.")

        # --- NEW: Implement stochastic multiple partitions as per Cluster-GCN paper ---
        group_size = self.config.PROTGRAM_CLUSTER_GROUP_SIZE
        if group_size > 1:
            print(f"  Stochastic multiple partitions enabled. Grouping {group_size} clusters per batch.")

        criterion = F.cross_entropy
        print(f"  Starting Cluster-GCN style training for up to {epochs} epochs on {len(node_partitions)} subgraphs (Task: {task_type})...")

        for epoch in range(1, epochs + 1):
            random.shuffle(node_partitions)

            # Group partitions into mini-batches as per the paper's strategy
            grouped_partitions = [
                node_partitions[i:i + group_size]
                for i in range(0, len(node_partitions), group_size)
            ]

            # --- NEW: Add accumulators for epoch-level metrics ---
            all_preds = []
            all_labels = []

            epoch_loss = 0.0
            optimizer.zero_grad()  # Zero gradients at the start of each epoch
            for i, partition_group in enumerate(tqdm(grouped_partitions, desc=f"  Epoch {epoch}", leave=False, disable=not self.config.DEBUG_VERBOSE)):
                # Create a self-contained PyG Data object for the current subgraph
                # --- DEFINITIVE FIX: Pass the homophily/heterophily matrices to the subgraph creator ---
                combined_node_indices = [node for partition in partition_group for node in partition]
                if not combined_node_indices: continue

                subgraph_data = full_data.graph_obj.create_subgraph_data_for_model(
                    model_type=model.__class__.__name__.lower(),
                    full_features=full_data.x,
                    full_labels=full_data.y,
                    node_subset=torch.tensor(combined_node_indices, dtype=torch.long),
                    # Pass the pre-calculated matrices from the full data object
                    A_homo_norm=getattr(full_data, 'A_homo_norm', None),
                    A_hetero_norm=getattr(full_data, 'A_hetero_norm', None)
                ).to(self.device)

                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp): # noqa
                    # --- REFACTOR: Use the centralized loss calculation helper ---
                    loss, preds, ground_truth = self._calculate_loss(
                        model, subgraph_data, criterion, task_type
                    )

                    # --- NEW: Store predictions and labels for epoch metrics ---
                    with torch.no_grad():
                        all_preds.append(preds.cpu())
                        all_labels.append(ground_truth.cpu())

                # --- Backward pass & Gradient Accumulation ---
                unnormalized_loss = loss.item()
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()

                # Update weights only after accumulating gradients for `accumulation_steps` batches
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(grouped_partitions):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                epoch_loss += unnormalized_loss

            # --- NEW: Calculate and log epoch-level metrics ---
            avg_epoch_loss = epoch_loss / len(grouped_partitions) if grouped_partitions else 0.0
            if self.config.DEBUG_VERBOSE and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
                current_lr = optimizer.param_groups[0]['lr']
                # Calculate accuracy from accumulated predictions
                train_acc = 0.0
                if all_preds and all_labels:
                    y_pred = torch.cat(all_preds).numpy()
                    y_true = torch.cat(all_labels).numpy()
                    if len(y_true) > 0:
                        train_acc = (y_pred == y_true).mean()

                log_str = (f"    Epoch: {epoch:03d}, Avg Batch Loss: {avg_epoch_loss:.4f}, "
                           f"Train Acc: {train_acc:.4f}, "
                           f"LR: {current_lr:.6f}")
                print(log_str)
            if scheduler: scheduler.step(avg_epoch_loss)
            if early_stopper and early_stopper.early_stop(avg_epoch_loss):
                print(f"  Early stopping triggered at epoch {epoch}. Best loss: {early_stopper.best_loss:.4f}")
                break

    def _load_graph_for_level(self, n: int) -> Optional[DirectedNgramGraph]:
        """
        Loads the pre-built graph for a specific n-gram level from disk.
        Includes an in-memory cache to avoid redundant loads.
        """
        if n in self._loaded_graphs:
            return self._loaded_graphs[n]

        graph_dir = self.config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{n}"
        if not graph_dir.exists():
            print(f"  ERROR: Graph directory for n={n} not found at '{graph_dir}'. Cannot proceed with this level.")
            return None

        print(f"  Loading graph for n={n} from: {graph_dir}")
        try:
            graph_obj = DirectedNgramGraph.load_from_dir(graph_dir)
            if graph_obj:
                self._loaded_graphs[n] = graph_obj
                return graph_obj
        except Exception as e:
            print(f"  ERROR: Failed to load graph for n={n}. Error: {e}")
            traceback.print_exc()
        return None

    def _get_initial_features_for_level(self, n: int, graph_obj: DirectedNgramGraph,
                                        prev_level_embeddings: Optional[np.ndarray],
                                        prev_level_map: Optional[Dict[str, int]]) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        Gets initial node features for a given n-gram level.
        - For n=1, features are initialized with an identity matrix.
        - For n>1, features are pooled from the (n-1) level embeddings.
        """
        if n == 1:
            print(f"  Initializing n=1 features with identity matrix.")
            features = torch.eye(graph_obj.number_of_nodes)
            return features, {}
        elif prev_level_embeddings is not None and prev_level_map is not None:
            return EmbeddingProcessor.pool_lower_level_embeddings_for_init(
                graph_obj, prev_level_embeddings, prev_level_map,
                strategy=self.config.PROTGRAM_HIERARCHICAL_POOLING_STRATEGY
            )
        else:
            print(f"  ERROR: Cannot initialize features for n={n}. Previous level embeddings or map are missing.")
            return None

    def _partition_graph(self, graph_obj: DirectedNgramGraph) -> List[List[int]]: # noqa
        """
        Partitions the graph for clustered training using the configured method.
        """
        method = self.config.PROTGRAM_PARTITIONING_METHOD
        print(f"  Partitioning graph with {graph_obj.number_of_nodes} nodes for clustered training (Method: {method})...")

        if method == 'graclus':
            from source.data_structures.coarsener import GraphCoarsener
            coarsening_level = self.config.PROTGRAM_COARSENING_LEVEL_FOR_PARTITIONING
            # --- DEFINITIVE FIX for AttributeError: Pass the correct graph object ---
            # The GraphCoarsener expects the custom DirectedNgramGraph object to access
            # its specific attributes like A_undirected_w.
            coarsening_result = GraphCoarsener.coarsen_graph(graph_obj, level=coarsening_level)
            if coarsening_result is None:
                print("  - WARNING: Graclus coarsening failed. Falling back to a single partition.")
                return [list(range(graph_obj.number_of_nodes))]

            coarsened_adj_index, cluster_map, coarsened_adj_weight = coarsening_result

            # --- NEW: Add optional validation step ---
            if self.config.PROTGRAM_VALIDATE_COARSENING:
                GraphCoarsener.validate_coarsening(
                    original_graph=graph_obj,
                    coarsened_edge_index=coarsened_adj_index,
                    coarsened_edge_weight=coarsened_adj_weight,
                    cluster_map=cluster_map
                )

            num_partitions = int(cluster_map.max().item()) + 1
            print(f"    Graclus algorithm found {num_partitions} communities.")

            # Group nodes by their partition ID from the cluster map tensor
            partitions = [[] for _ in range(num_partitions)]
            for node_idx, cluster_id in enumerate(cluster_map.tolist()):
                partitions[cluster_id].append(node_idx)

        elif method == 'louvain':
            import community as community_louvain
            import networkx as nx
            # Use the undirected, unweighted graph for community detection as it's standard.
            if graph_obj.A_undirected_norm_sparse is None or graph_obj.A_undirected_norm_sparse._nnz() == 0:
                print("  - WARNING: Undirected matrix not available for Louvain. Returning single partition.")
                return [list(range(graph_obj.number_of_nodes))]

            edge_index = graph_obj.A_undirected_norm_sparse.indices().cpu().numpy()
            G_nx = nx.Graph()
            G_nx.add_nodes_from(range(graph_obj.number_of_nodes))
            G_nx.add_edges_from(edge_index.T)

            partition_map = community_louvain.best_partition(G_nx, random_state=self.config.RANDOM_STATE)
            num_partitions = len(set(partition_map.values()))
            print(f"    Louvain algorithm found {num_partitions} communities.")

            # Group nodes by their partition ID
            partitions = [[] for _ in range(num_partitions)]
            for node, part_id in partition_map.items():
                partitions[part_id].append(node)
        else:
            raise ValueError(f"Unknown partitioning method: '{method}'")

        print(f"    Created {len(partitions)} partitions.")
        return [p for p in partitions if p]  # Return non-empty partitions

    def _get_level_ngram_maps(self) -> Dict[int, Dict[str, int]]:
        """Returns the node_to_idx maps for all loaded graph levels."""
        return {n: graph.get_node_to_idx_map() for n, graph in self._loaded_graphs.items()}

    def _load_graph_for_level(self, n: int) -> Optional[DirectedNgramGraph]:
        """
        Loads the pre-built graph for a specific n-gram level from disk.
        Includes an in-memory cache to avoid redundant loads.
        """
        if n in self._loaded_graphs:
            return self._loaded_graphs[n]

        graph_dir = self.config.RESULTS_GRAPH_OBJECTS_DIR / f"ngram_graph_n{n}"
        if not graph_dir.exists():
            print(f"  ERROR: Graph directory for n={n} not found at '{graph_dir}'. Cannot proceed with this level.")
            return None

        print(f"  Loading graph for n={n} from: {graph_dir}")
        try:
            graph_obj = DirectedNgramGraph.load_from_dir(graph_dir)
            if graph_obj:
                self._loaded_graphs[n] = graph_obj
                return graph_obj
        except Exception as e:
            print(f"  ERROR: Failed to load graph for n={n}. Error: {e}")
            traceback.print_exc()
        return None

    def _get_initial_features_for_level(self, n: int, graph_obj: DirectedNgramGraph,
                                        prev_level_embeddings: Optional[np.ndarray],
                                        prev_level_map: Optional[Dict[str, int]]) -> Optional[Tuple[torch.Tensor, Dict]]:
        """
        Gets initial node features for a given n-gram level.
        - For n=1, features are randomly initialized.
        - For n>1, features are pooled from the (n-1) level embeddings.
        """
        if n == 1:
            print(f"  Initializing n=1 features with random noise (dim={self.config.PROTGRAM_1GRAM_INIT_DIM}).")
            features = torch.randn((graph_obj.number_of_nodes, self.config.PROTGRAM_1GRAM_INIT_DIM))
            return features, {}
        elif prev_level_embeddings is not None and prev_level_map is not None:
            return EmbeddingProcessor.pool_lower_level_embeddings_for_init(
                graph_obj, prev_level_embeddings, prev_level_map,
                strategy=self.config.PROTGRAM_HIERARCHICAL_POOLING_STRATEGY
            )
        else:
            print(f"  ERROR: Cannot initialize features for n={n}. Previous level embeddings or map are missing.")
            return None