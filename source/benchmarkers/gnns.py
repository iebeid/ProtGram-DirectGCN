# ==============================================================================
# MODULE: benchmarkers/gnns.py
# PURPOSE: Handles benchmarking of various GNN models on standard datasets.
# VERSION: 5.0 (Refactored to use BaseBenchmarker)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import traceback
from typing import Dict, List, Any, Tuple, Optional
import mlflow
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch_geometric.data import Data
from torch_geometric.utils import homophily

from configuration.config import Config
from source.benchmarkers.base import BaseBenchmarker
from source.utils.fs.file_utils import FileUtils
from source.models.factory import ModelFactory
from source.data_structures.direct_ngram_graph import DirectedNgramGraph
from source.utils.data.data_utils import DataUtils
# --- DEFINITIVE FIX: Import the Network Embedding Benchmarker ---
from source.benchmarkers.nes import NetworkEmbeddingBenchmarker


class GNNBenchmarker(BaseBenchmarker):
    def __init__(self, config: Config):
        super().__init__(config, "GNN Benchmarker")
        self.embedding_dir = config.RESULTS_BENCHMARK_EMBEDDINGS_DIR
        self.model_factory = ModelFactory(config, context='benchmark')
        print(f"Benchmark embeddings will be saved to: {self.embedding_dir}")

    def _save_embeddings(self, model: torch.nn.Module, data: Data):
        """Extracts and saves GNN node embeddings to an H5 file."""
        print(f"    Extracting embeddings for {model.__class__.__name__}...")
        with torch.no_grad():
            model.eval()
            _, embeddings = model(data)
        if embeddings is None:
            print("    Warning: Could not extract embeddings.")
            return

        embeddings_np = embeddings.cpu().numpy()
        emb_dict = {str(i): embeddings_np[i] for i in range(embeddings_np.shape[0])}

        # Construct the full path for the output file
        dataset_name = getattr(data, 'name', 'unknown_dataset')
        model_name = model.__class__.__name__
        variant_suffix = getattr(data, 'variant_suffix', None)
        emb_dim = embeddings_np.shape[1]

        base_name = f"{model_name}_embeddings_dim{emb_dim}" if not variant_suffix else f"{model_name}_{variant_suffix}_embeddings_dim{emb_dim}"
        h5_filename = f"{base_name}.h5"
        full_h5_path = self.embedding_dir / dataset_name / h5_filename

        FileUtils.write_h5(emb_dict, full_h5_path, f"Writing H5 for {model_name}")
        print(f"      Saved embeddings to {full_h5_path}")

    def _train_and_evaluate(self, model: torch.nn.Module, data: Data,
                            epochs_override: Optional[int] = None,
                            lr_override: Optional[float] = None,
                            wd_override: Optional[float] = None) -> Tuple[Dict[str, float], pd.DataFrame]:
        """Handles the training and evaluation loop for a given model and data."""
        model.to(self.device)
        data = data.to(self.device)
        lr = lr_override if lr_override is not None else self.config.BENCHMARK_GNN_LEARNING_RATE
        wd = wd_override if wd_override is not None else self.config.BENCHMARK_GNN_WEIGHT_DECAY
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        best_val_acc = -1
        test_acc_at_best_val = -1
        # --- DEFINITIVE FIX for UnboundLocalError ---
        # Initialize metrics to a default value before the loop.
        f1_at_best_val = -1.0
        precision_at_best_val = -1.0
        recall_at_best_val = -1.0
        history = {'epoch': [], 'loss': [], 'val_acc': [], 'test_acc': []}

        train_mask = self._get_1d_mask(data.train_mask)
        val_mask = self._get_1d_mask(data.val_mask)
        test_mask = self._get_1d_mask(data.test_mask)

        total_epochs = int(epochs_override) if epochs_override is not None else int(self.config.BENCHMARK_GNN_EPOCHS)
        for epoch in range(1, total_epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(data)
            target = data.y[train_mask].long()  # Ensure target is Long type
            loss = F.cross_entropy(logits[train_mask], target)
            loss.backward()
            # --- FIX: Add Gradient Clipping to stabilize training on small/volatile graphs ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                logits_eval, _ = model(data)
                pred = logits_eval.argmax(dim=1)
                val_correct = (pred[val_mask] == data.y[val_mask]).sum()
                val_acc = int(val_correct) / int(val_mask.sum()) if val_mask.sum() > 0 else 0.0
                test_correct = (pred[test_mask] == data.y[test_mask]).sum()
                test_acc = int(test_correct) / int(test_mask.sum()) if test_mask.sum() > 0 else 0.0

            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['val_acc'].append(val_acc)
            history['test_acc'].append(test_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val = test_acc
                # --- NEW: Capture all test metrics at the best validation epoch ---
                y_true_test = data.y[test_mask].cpu().numpy()
                y_pred_test = pred[test_mask].cpu().numpy()
                f1_at_best_val = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)
                precision_at_best_val = precision_score(y_true_test, y_pred_test, average='macro', zero_division=0)
                recall_at_best_val = recall_score(y_true_test, y_pred_test, average='macro', zero_division=0)

            if self.config.DEBUG_VERBOSE and (epoch == 1 or epoch % 10 == 0 or epoch == self.config.BENCHMARK_GNN_EPOCHS):
                print(f"    Epoch {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

        metrics = {
            'Accuracy': test_acc_at_best_val,
            'F1-Score (Macro)': f1_at_best_val,
            'Precision (Macro)': precision_at_best_val,
            'Recall (Macro)': recall_at_best_val
        }

        print(f"  Finished training. Best Val Acc: {best_val_acc:.4f}, Corresponding Test Acc: {test_acc_at_best_val:.4f}")

        if self.config.BENCHMARK_SAVE_EMBEDDINGS:
            self._save_embeddings(model, data)

        return metrics, pd.DataFrame(history)

    def _prepare_data_for_model(self, model_name: str, data: Data, is_heterophilic: bool) -> Data:
        """
        A just-in-time data preparation utility. It creates a model-specific
        data object with the correct graph representations.
        """
        # Start with a fresh clone of the original data
        data_for_model = data.clone()
        model_name_lower = model_name.lower()

        # --- Base representation: Raw undirected graph ---
        # Most models (GCN, GAT, etc.) will use this and apply their own normalization.
        base_edge_weight = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else torch.ones(data.edge_index.shape[1], device=data.edge_index.device)
        A_out_w_sparse = torch.sparse_coo_tensor(data.edge_index, base_edge_weight, (data.num_nodes, data.num_nodes)).coalesce()
        A_undir_w = (A_out_w_sparse + A_out_w_sparse.t()).coalesce()
        data_for_model.edge_index = A_undir_w.indices()
        data_for_model.edge_attr = A_undir_w.values()

        # --- Model-specific overrides ---
        if model_name_lower == 'directgcn':
            print("    -> Preparing specialized matrices for DirectGCN...")
            # --- REFACTOR: Use the static methods from DirectedNgramGraph ---
            A_in_w_sparse = A_out_w_sparse.t().coalesce()
            data_for_model.mathcal_A_in = DirectedNgramGraph._calculate_single_propagation_matrix(
                A_in_w_sparse, data.num_nodes, self.config.GCN_PROPAGATION_EPSILON)
            data_for_model.mathcal_A_out = DirectedNgramGraph._calculate_single_propagation_matrix(
                A_out_w_sparse, data.num_nodes, self.config.GCN_PROPAGATION_EPSILON)
            data_for_model.A_undirected_norm_sparse = DirectedNgramGraph._normalize_symmetric_matrix(
                A_undir_w, data.num_nodes)

            # Attach as edge_index attributes for compatibility with the model's forward pass
            data_for_model.edge_index_mathcal_in = data_for_model.mathcal_A_in.indices()
            data_for_model.edge_weight_mathcal_in = data_for_model.mathcal_A_in.values()
            data_for_model.edge_index_mathcal_out = data_for_model.mathcal_A_out.indices()
            data_for_model.edge_weight_mathcal_out = data_for_model.mathcal_A_out.values()
            data_for_model.edge_index_undirected_norm = data_for_model.A_undirected_norm_sparse.indices()
            data_for_model.edge_weight_undirected_norm = data_for_model.A_undirected_norm_sparse.values()

            if is_heterophilic:
                A_homo_norm, A_hetero_norm = DirectedNgramGraph.split_edges_by_homophily(A_out_w_sparse, data.num_nodes, data.y)
                data_for_model.edge_index_homo_norm, data_for_model.edge_weight_homo_norm = A_homo_norm.indices(), A_homo_norm.values()
                data_for_model.edge_index_hetero_norm, data_for_model.edge_weight_hetero_norm = A_hetero_norm.indices(), A_hetero_norm.values()

        elif model_name_lower == 'rgcn':
            print("    -> Preparing directed edge format for RGCN.")
            edge_index_out = A_out_w_sparse.indices()
            edge_index_in = A_out_w_sparse.t().coalesce().indices()
            data_for_model.edge_index = torch.cat([edge_index_out, edge_index_in], dim=1)
            edge_type_out = torch.zeros(edge_index_out.size(1), dtype=torch.long, device=edge_index_out.device)
            edge_type_in = torch.ones(edge_index_in.size(1), dtype=torch.long, device=edge_index_in.device)
            data_for_model.edge_type = torch.cat([edge_type_out, edge_type_in])
            if 'edge_attr' in data_for_model:
                del data_for_model.edge_attr

        elif model_name_lower == 'dirgnn':
            print("    -> Preparing raw directed edge format for DirGNN.")
            data_for_model.edge_index = A_out_w_sparse.indices()
            data_for_model.edge_attr = A_out_w_sparse.values()
            data_for_model.edge_index_backward = A_out_w_sparse.t().coalesce().indices()

        return data_for_model

    def _run_on_dataset_variant(self, dataset: Any, variant_name: str) -> List[Dict]:
        """Runs all configured models on a single dataset variant."""
        print(f"\n" + "=" * 50)
        print(f"### Benchmarking on Dataset: {variant_name} ###")
        print("=" * 50 + "\n")

        data = dataset[0]
        data.name = variant_name

        # --- FIX for CUDA device-side assert ---
        # Instead of trusting dataset.num_classes, derive it directly from the labels.
        # This prevents errors if labels are e.g., [1, 2, 3, 4] but num_classes is reported as 4.
        num_classes = int(data.y.max().item()) + 1
        # --- END FIX ---

        # Handle data splits
        if not all(hasattr(data, mask) and getattr(data, mask) is not None and getattr(data, mask).any() for mask in ['train_mask', 'val_mask', 'test_mask']):
            print(f"  Generating custom seeded split for {variant_name}.")
            num_nodes = data.num_nodes
            # FIX: Use a seeded random number generator for reproducible splits.
            rng = np.random.default_rng(self.config.RANDOM_STATE)
            indices = rng.permutation(num_nodes)
            train_size = int(num_nodes * self.config.BENCHMARK_SPLIT_RATIOS['train'])
            val_size = int(num_nodes * self.config.BENCHMARK_SPLIT_RATIOS['val'])
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.train_mask[indices[:train_size]] = True
            data.val_mask[indices[train_size:train_size + val_size]] = True
            data.test_mask[indices[train_size + val_size:]] = True
        else:
            print(f"  Using existing standard masks for {variant_name}.")

        print(f"  {variant_name.split('_')[0]} loaded: Nodes={data.num_nodes}, Edges={data.num_edges}, Features={data.num_features}, Classes={num_classes}")

        # --- NEW: Ignore existing features and initialize random features ---
        # This aligns the benchmark with the ProtGram n=1 setup, testing the
        # models' ability to learn from structure alone without relying on
        # pre-existing node attributes.
        new_feature_dim = self.config.BENCHMARK_GNN_INIT_DIM
        print(f"  Ignoring original features. Initializing new random features with dimension: {new_feature_dim}")
        # Create the random features on the CPU; they will be moved to the GPU later.
        data.x = torch.randn((data.num_nodes, new_feature_dim))
        # The data.num_features property will now automatically reflect the new dimension.

        # --- NEW: Dynamic Architecture Selection for DirectGCN ---
        # Calculate the graph's homophily to decide whether to use the specialized paths.
        # This allows the model to adapt to the dataset's characteristics.
        homophily_ratio = homophily(data.edge_index, data.y, method='edge')
        # A common threshold is 0.6. Below this, the graph is considered heterophilic.
        is_heterophilic = homophily_ratio < self.config.GCN_HETEROPHILY_THRESHOLD
        print(f"  Dataset Homophily Ratio: {homophily_ratio:.4f}. Is Heterophilic? -> {is_heterophilic}")

        results = []
        for model_name in self.config.BENCHMARK_GNN_MODELS_TO_RUN:
            print(f"\n--- Benchmarking Model: {model_name} on Dataset: {variant_name} ---")
            try:
                if model_name.lower() == 'directgcn':
                    # Small ablation sweep with tuned hyperparameters
                    base_lr = float(self.config.BENCHMARK_GNN_LEARNING_RATE)
                    base_wd = float(self.config.BENCHMARK_GNN_WEIGHT_DECAY)
                    base_epochs = int(self.config.BENCHMARK_GNN_EPOCHS)
                    try:
                        base_dropout = float(self.config.BENCHMARK_GNN_DROPOUT_RATE)
                    except Exception:
                        base_dropout = 0.5

                    variants = [
                        {'suffix': 'full_scalar', 'gating_mode': 'scalar', 'path_selection': 'full',
                         'dropout': min(0.3, base_dropout), 'lr': base_lr * 0.5, 'wd': base_wd * 2.0, 'epochs': base_epochs * 2},
                        {'suffix': 'undir_none', 'gating_mode': 'none', 'path_selection': 'undirected',
                         'dropout': min(0.3, base_dropout), 'lr': base_lr * 0.5, 'wd': base_wd * 2.0, 'epochs': base_epochs * 2},
                        {'suffix': 'inout_scalar', 'gating_mode': 'scalar', 'path_selection': 'in_out',
                         'dropout': min(0.3, base_dropout), 'lr': base_lr * 0.5, 'wd': base_wd * 2.0, 'epochs': base_epochs * 2},
                    ]

                    for var in variants:
                        variant_name_tag = f"{model_name}[{var['suffix']}]"
                        with mlflow.start_run(run_name=f"{variant_name_tag}_on_{variant_name}", nested=True):
                            mlflow.set_tag("model_name", variant_name_tag)
                            mlflow.set_tag("dataset_name", variant_name)
                            mlflow.log_param("epochs", var['epochs'])
                            mlflow.log_param("learning_rate", var['lr'])
                            mlflow.log_param("weight_decay", var['wd'])
                            mlflow.log_param("gating_mode", var['gating_mode'])
                            mlflow.log_param("path_selection", var['path_selection'])
                            mlflow.log_param("is_undirected", "_Undirected" in variant_name)

                            data_for_model = self._prepare_data_for_model(
                                model_name=model_name,
                                data=data,
                                is_heterophilic=is_heterophilic
                            )
                            # mark variant for embedding filenames
                            setattr(data_for_model, 'variant_suffix', var['suffix'])

                            model = self.model_factory.create_model(
                                model_name=model_name, in_channels=data_for_model.num_features,
                                num_classes=num_classes, num_graph_nodes=data.num_nodes,
                                use_homo_heterophilic=False if not is_heterophilic else True,  # typo-safe
                                use_homo_hetero_paths=is_heterophilic,
                                gating_mode=var['gating_mode'],
                                dropout_rate=var['dropout'],
                                path_selection=var['path_selection']
                            )

                            if self.config.DEBUG_VERBOSE:
                                print("  Variant Architecture:")
                                print(model)

                            metrics, history_df = self._train_and_evaluate(
                                model, data_for_model,
                                epochs_override=var['epochs'],
                                lr_override=var['lr'],
                                wd_override=var['wd']
                            )
                            # Keep base model name for summary categorization; store variant separately
                            result_row = {
                                "dataset": variant_name,
                                "model": model_name,
                                "variant": var['suffix'],
                                "error": None
                            }
                            result_row.update(metrics)
                            results.append(result_row)

                            mlflow.log_metrics({
                                "test_accuracy": metrics.get('Accuracy', 0.0),
                                "f1_macro": metrics.get('F1-Score (Macro)', 0.0)
                            })
                            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
                            history_path = Path(self.output_dir) / f"history_{variant_name_tag}_{variant_name}.csv"
                            history_df.to_csv(history_path, index=False)
                            mlflow.log_artifact(str(history_path), "training_history")
                            if os.path.exists(history_path):
                                os.remove(history_path)
                else:
                    with mlflow.start_run(run_name=f"{model_name}_on_{variant_name}", nested=True):
                        mlflow.set_tag("model_name", model_name)
                        mlflow.set_tag("dataset_name", variant_name)
                        mlflow.log_param("epochs", self.config.BENCHMARK_GNN_EPOCHS)
                        mlflow.log_param("learning_rate", self.config.BENCHMARK_GNN_LEARNING_RATE)
                        mlflow.log_param("is_undirected", "_Undirected" in variant_name)

                        data_for_model = self._prepare_data_for_model(
                            model_name=model_name,
                            data=data,
                            is_heterophilic=is_heterophilic
                        )

                        model = self.model_factory.create_model(
                            model_name=model_name, in_channels=data_for_model.num_features,
                            num_classes=num_classes, num_graph_nodes=data.num_nodes,
                            use_homo_hetero_paths=is_heterophilic
                        )

                        if self.config.DEBUG_VERBOSE:
                            print("  Model Architecture:")
                            print(model)

                        metrics, history_df = self._train_and_evaluate(model, data_for_model)
                        result_row = {"dataset": variant_name, "model": model_name, "error": None}
                        result_row.update(metrics)
                        results.append(result_row)

                        mlflow.log_metrics({
                            "test_accuracy": metrics.get('Accuracy', 0.0),
                            "f1_macro": metrics.get('F1-Score (Macro)', 0.0)
                        })
                        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
                        history_path = Path(self.output_dir) / f"history_{model_name}_{variant_name}.csv"
                        history_df.to_csv(history_path, index=False)
                        mlflow.log_artifact(str(history_path), "training_history")
                        if os.path.exists(history_path):
                            os.remove(history_path)

            except Exception as e:
                print(f"ERROR during training/evaluation of {model_name} on {variant_name}: {e}")
                traceback.print_exc()
                mlflow.set_tag("status", "FAILED")
                mlflow.log_param("error", str(e))
                results.append({"dataset": variant_name, "model": model_name, "error": str(e)})
        return results

    def run(self):
        """Main execution function for the benchmarker."""
        DataUtils.print_header("PIPELINE: GNN BENCHMARKER")
        all_results = []
        print(f"Standard PyG datasets will be stored in/loaded from: {self.dataset_root}")

        for dataset_name in self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            dataset_results = []

            # --- Run on Original (potentially directed) Graph ---
            dataset_original = self._get_dataset(dataset_name)
            if dataset_original:
                dataset_results.extend(self._run_on_dataset_variant(dataset_original, f"{dataset_name}_Original"))

            # --- DEFINITIVE FIX: Implement the logic for the BENCHMARK_TEST_ON_UNDIRECTED flag ---
            # This was a missing feature. The pipeline will now run a second evaluation
            # on a strictly undirected version of the graph if the flag is set.
            if self.config.BENCHMARK_TEST_ON_UNDIRECTED and dataset_original:
                from torch_geometric.utils import to_undirected
                data_undirected = dataset_original[0].clone()
                # Create a truly undirected graph by removing self-loops and adding reverse edges
                data_undirected.edge_index = to_undirected(data_undirected.edge_index, data_undirected.num_nodes)
                dataset_results.extend(self._run_on_dataset_variant([data_undirected], f"{dataset_name}_Undirected"))

            # --- Save summary for the current dataset ---
            if dataset_results:
                all_results.extend(dataset_results)

        # --- Now, run the Network Embedding benchmarks as a subroutine ---
        print("\n--- Launching Network Embedding Benchmark Sub-routine ---")
        ne_benchmarker = NetworkEmbeddingBenchmarker(self.config)
        ne_results_df = ne_benchmarker.run()
        if ne_results_df is not None and not ne_results_df.empty:
            # Convert DataFrame to list of records and extend the main results list
            all_results.extend(ne_results_df.to_dict('records'))

        # --- Save a final, grand summary of all results ---
        if all_results:
            summary_df = pd.DataFrame(all_results)
            # Save grand summary under the configured benchmarking directory
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            summary_path = Path(self.output_dir) / "summary.csv"
            try:
                summary_df.to_csv(summary_path, index=False)
                print(f"  Benchmarking summary saved to: {summary_path}")
            except Exception as e:
                print(f"  WARNING: Could not save benchmarking summary: {e}")

            DataUtils.print_header("GNN Benchmarking PIPELINE FINISHED")
            # --- FIX: Explicitly clean up resources to prevent hangs before user prompts ---
            del all_results
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # --- END FIX ---
            return summary_df
        return pd.DataFrame()