# ==============================================================================
# MODULE: gnn_benchmarker.py
# PURPOSE: To benchmark various GNN models on standard datasets.
# VERSION: 3.4.18 (Definitive fix for mask attribute and data loading)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import random
import time
from typing import Dict, List, Tuple, Any, Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.data import Data
from torch_geometric.datasets import (KarateClub, Planetoid)
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import add_self_loops, degree, to_undirected

from config import Config
from src.models.gnn_zoo import *
from src.models.protgram_directgcn import ProtGramDirectGCN
from src.utils.data_utils import DataUtils
from src.utils.models_utils import EmbeddingProcessor


class GNNBenchmarker:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(str(self.config.BENCHMARK_EMBEDDINGS_DIR), exist_ok=True)
        print(f"GNNBenchmarker initialized. Using device: {self.device}")
        print(f"Benchmark embeddings will be saved to: {self.config.BENCHMARK_EMBEDDINGS_DIR}")

    @staticmethod
    def set_seeds(seed: int):
        """Sets random seeds for reproducibility across all relevant libraries."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # for multi-GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"  Seeds set to {seed} for reproducibility.")

    def _get_dataset(self, name: str, root_path: str, make_undirected: bool = False):
        print(f"  Attempting to load dataset: {name} (root: {root_path}, undirected_requested: {make_undirected})...")

        dataset_obj = None
        if name.lower() == 'karateclub':
            dataset_obj = KarateClub(transform=None)
        elif name.lower() in ['cora', 'citeseer', 'pubmed']:
            dataset_obj = Planetoid(root=root_path, name=name, transform=None)
        elif name.lower() in ['cornell', 'texas', 'wisconsin']:
            from torch_geometric.datasets import WebKB
            dataset_obj = WebKB(root=root_path, name=name, transform=None)
        else:
            print(f"ERROR: Dataset '{name}' loader not implemented.")
            return None, None, None, None

        raw_data_obj = dataset_obj[0]

        # Extract core attributes and ensure they are on CPU initially
        x = raw_data_obj.x.cpu() if raw_data_obj.x is not None else torch.empty(raw_data_obj.num_nodes, dataset_obj.num_features)
        edge_index = raw_data_obj.edge_index.cpu()
        y = raw_data_obj.y.cpu() if raw_data_obj.y is not None else torch.empty(raw_data_obj.num_nodes, dtype=torch.long)
        num_nodes = raw_data_obj.num_nodes

        if make_undirected:
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        # --- DEFINITIVE FIX for AttributeError: Check for ALL masks before using them ---
        use_existing_masks = (
            hasattr(raw_data_obj, 'train_mask') and hasattr(raw_data_obj, 'val_mask') and hasattr(raw_data_obj, 'test_mask') and
            raw_data_obj.train_mask is not None and raw_data_obj.val_mask is not None and raw_data_obj.test_mask is not None and
            raw_data_obj.train_mask.ndim == 1
        )

        if use_existing_masks:
            print(f"  Using existing standard masks for {name}.")
            train_mask = raw_data_obj.train_mask.cpu()
            val_mask = raw_data_obj.val_mask.cpu()
            test_mask = raw_data_obj.test_mask.cpu()
        else:
            print(f"  Generating custom seeded split for {name}.")
            g = torch.Generator().manual_seed(self.config.RANDOM_STATE)
            indices = torch.randperm(num_nodes, generator=g)
            num_train = int(self.config.BENCHMARK_SPLIT_RATIOS['train'] * num_nodes)
            num_val = int(self.config.BENCHMARK_SPLIT_RATIOS['val'] * num_nodes)

            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[indices[:num_train]] = True
            val_mask[indices[num_train:num_train + num_val]] = True
            test_mask[indices[num_train + num_val:]] = True
            print(f"  Applied custom seeded split. Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
        # --- END FIX ---

        # Create a brand new Data object with all attributes explicitly passed
        final_data_obj = Data(x=x, edge_index=edge_index, y=y,
                              train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
                              num_nodes=num_nodes)

        # Now apply the ToDevice transform to the fully constructed Data object
        transform_compose = T.Compose([T.ToDevice(self.device)])
        final_data_obj = transform_compose(final_data_obj)

        print(f"  {name} loaded: Nodes: {final_data_obj.num_nodes}, Edges: {final_data_obj.num_edges}, Features: {dataset_obj.num_features}, Classes: {dataset_obj.num_classes}")
        return final_data_obj, final_data_obj, final_data_obj, 'single-label-node'

    def _get_undirected_normalized_edges(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = data.num_nodes
        edge_index = data.edge_index.to(self.device).long()

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            return torch.empty((2, 0), dtype=torch.long, device=self.device), torch.empty((0,), dtype=torch.float32, device=self.device)

        undir_edge_index_raw, _ = to_undirected(edge_index, num_nodes=num_nodes)

        if undir_edge_index_raw.ndim != 2 or undir_edge_index_raw.shape[0] != 2:
            if undir_edge_index_raw.ndim == 1 and undir_edge_index_raw.numel() > 0 and undir_edge_index_raw.numel() % 2 == 0:
                print(f"WARNING: to_undirected returned unexpected 1D shape {undir_edge_index_raw.shape}. Attempting to reshape to (2, N/2).")
                undir_edge_index = undir_edge_index_raw.reshape(2, -1)
            else:
                print(f"WARNING: to_undirected returned unexpected shape {undir_edge_index_raw.shape}. Converting to (2, 0) empty tensor.")
                undir_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        else:
            undir_edge_index = undir_edge_index_raw

        undir_edge_index, _ = add_self_loops(undir_edge_index, num_nodes=num_nodes)

        if undir_edge_index.numel() == 0:
            return undir_edge_index, torch.empty((0,), dtype=torch.float32, device=self.device)

        edge_weight = torch.ones(undir_edge_index.size(1), dtype=torch.float32, device=self.device)
        row, col = undir_edge_index
        deg = degree(col, num_nodes=num_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return undir_edge_index, norm_values

    def _save_node_embeddings(self, model: nn.Module, data_for_emb, model_name: str, dataset_name: str, graph_variant_suffix: str):
        if not self.config.BENCHMARK_SAVE_EMBEDDINGS:
            return

        print(f"    Extracting embeddings for {model_name} on {dataset_name}{graph_variant_suffix}...")
        model.eval()
        with torch.no_grad():
            node_embeddings_tensor = None
            if isinstance(data_for_emb, Data):
                if model_name == "ProtGramDirectGCN":
                    _, node_embeddings_tensor = model(data=data_for_emb)
                elif hasattr(model, 'get_embeddings') and callable(getattr(model, 'get_embeddings')):
                    node_embeddings_tensor = model.get_embeddings(data_for_emb)
                elif hasattr(model, 'embedding_output') and model.embedding_output is not None:
                    node_embeddings_tensor = model.embedding_output
                else:
                    output = model(data_for_emb)
                    if isinstance(output, tuple):
                        node_embeddings_tensor = output[1] if len(output) > 1 and output[1] is not None else output[0]
                    else:
                        node_embeddings_tensor = output
                    if node_embeddings_tensor.shape[-1] == data_for_emb.y.max().item() + 1:
                        print(f"      Using final layer output (likely logits) as embeddings for {model_name}.")

                if node_embeddings_tensor is None:
                    print(f"      Could not extract embeddings for {model_name}. Skipping save.")
                    return
                node_embeddings_np = node_embeddings_tensor.cpu().numpy()
            else:
                print(f"      Unsupported data type for embedding extraction: {type(data_for_emb)}. Skipping save.")
                return

        if node_embeddings_np.size == 0:
            print(f"      Embeddings are empty for {model_name}. Skipping save.")
            return

        emb_dim = node_embeddings_np.shape[1]
        filename_suffix = f"_dim{emb_dim}"

        if self.config.BENCHMARK_APPLY_PCA_TO_EMBEDDINGS and emb_dim > self.config.BENCHMARK_PCA_TARGET_DIM and node_embeddings_np.shape[0] > self.config.BENCHMARK_PCA_TARGET_DIM:
            print(f"      Applying PCA to {model_name} embeddings (target dim: {self.config.BENCHMARK_PCA_TARGET_DIM})...")
            temp_emb_dict = {str(i): node_embeddings_np[i] for i in range(node_embeddings_np.shape[0])}
            pca_embeds_dict = EmbeddingProcessor.apply_pca(temp_emb_dict, self.config.BENCHMARK_PCA_TARGET_DIM, self.config.RANDOM_STATE, output_dtype=np.float32)
            if pca_embeds_dict:
                pca_node_embeddings_np = np.array([pca_embeds_dict[str(i)] for i in range(len(pca_embeds_dict))])
                if pca_node_embeddings_np.size > 0:
                    node_embeddings_np = pca_node_embeddings_np
                    emb_dim = node_embeddings_np.shape[1]
                    filename_suffix = f"_pca{emb_dim}"
                    print(f"      PCA applied. New embedding dim: {emb_dim}")
                else:
                    print(f"      PCA resulted in empty embeddings for {model_name}, using full dimension embeddings.")
            else:
                print(f"      PCA failed or was skipped for {model_name}, using full dimension embeddings.")

        emb_dir = os.path.join(str(self.config.BENCHMARK_EMBEDDINGS_DIR), f"{dataset_name}{graph_variant_suffix}")
        os.makedirs(emb_dir, exist_ok=True)
        emb_filename = f"{model_name}_embeddings{filename_suffix}.h5"
        emb_path = os.path.join(emb_dir, emb_filename)

        try:
            with h5py.File(emb_path, 'w') as hf:
                hf.create_dataset('embeddings', data=node_embeddings_np)
                hf.attrs['model_name'] = model_name
                hf.attrs['dataset_name'] = dataset_name
                hf.attrs['graph_variant'] = graph_variant_suffix
                hf.attrs['shape'] = node_embeddings_np.shape
            print(f"      Saved {model_name} embeddings for {dataset_name}{graph_variant_suffix} to {emb_path}")
        except Exception as e:
            print(f"      ERROR saving embeddings to {emb_path}: {e}")

    def train_and_evaluate(
            self, model_name_str: str, model: nn.Module, dataset_name: str,
            train_data, val_data, test_data,
            optimizer: torch.optim.Optimizer, num_epochs: int, task_type: str,
            num_classes: int):

        best_val_metric = 0.0
        best_test_metric_at_best_val = 0.0
        history = []
        final_trained_model_state = None

        model.to(self.device)
        print(f"  Training {model_name_str} on {dataset_name} using device: {self.device} for {num_epochs} epochs.")

        criterion = F.cross_entropy
        metric_name = "Accuracy"
        metric_fn = lambda y_true, y_pred_logits: accuracy_score(y_true.cpu().numpy(), y_pred_logits.detach().argmax(dim=-1).cpu().numpy())
        print(f"  Using Loss: cross_entropy, Metric: {metric_name}")

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            if model_name_str == "ProtGramDirectGCN":
                out, _ = model(data=train_data)
            else:
                out = model(train_data)

            loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                if model_name_str == "ProtGramDirectGCN":
                    out_val, _ = model(data=val_data)
                else:
                    out_val = model(val_data)
                val_metric = metric_fn(val_data.y[val_data.val_mask], out_val[val_data.val_mask])

            history.append({'epoch': epoch, 'loss': loss.item(), f'val_{metric_name.lower()}': val_metric})

            if val_metric >= best_val_metric:
                best_val_metric = val_metric
                final_trained_model_state = model.state_dict()
                with torch.no_grad():
                    if model_name_str == "ProtGramDirectGCN":
                        out_test, _ = model(data=test_data)
                    else:
                        out_test = model(test_data)
                    best_test_metric_at_best_val = metric_fn(test_data.y[test_data.test_mask], out_test[test_data.test_mask])

        if epoch == num_epochs - 1:
            print(f"    Epoch {epoch:03d}, Loss: {loss.item():.4f}, Val {metric_name}: {val_metric:.4f}")

        print(f"  Finished training for {model_name_str} on {dataset_name}.")
        print(f"  Best Val {metric_name}: {best_val_metric:.4f}, Corresponding Test {metric_name}: {best_test_metric_at_best_val:.4f}")

        if final_trained_model_state:
            model.load_state_dict(final_trained_model_state)

        return best_val_metric, best_test_metric_at_best_val, pd.DataFrame(history), metric_name

    def run_on_dataset_variant(self, dataset_name: str, model_zoo_config: Dict,
                               train_data, val_data, test_data, task_type: str,
                               num_features: int, num_classes: int, graph_variant_suffix: str = ""):
        dataset_results = []
        DataUtils.print_header(f"Benchmarking on Dataset: {dataset_name}{graph_variant_suffix}")

        pgd_train_data, pgd_val_data, pgd_test_data = None, None, None
        try:
            print(f"\n--- Pre-processing data for ProtGramDirectGCN on {dataset_name}{graph_variant_suffix} ---")
            pgd_train_data, pgd_val_data, pgd_test_data = [d.clone() for d in [train_data, val_data, test_data]]
            for d in [pgd_train_data, pgd_val_data, pgd_test_data]:
                d.edge_index_out = d.edge_index
                d.edge_weight_out = getattr(d, 'edge_attr', None)
                d.edge_index_in = d.edge_index[[1, 0]]
                d.edge_weight_in = getattr(d, 'edge_attr', None)
                undir_idx, undir_w = self._get_undirected_normalized_edges(d)
                d.edge_index_undirected_norm = undir_idx
                d.edge_weight_undirected_norm = undir_w
            print("--- Pre-processing complete ---")
        except Exception as e:
            print(f"ERROR during pre-processing for ProtGramDirectGCN: {e}")
            import traceback
            traceback.print_exc()

        all_models_to_run = list(model_zoo_config.keys()) + ["ProtGramDirectGCN"]
        for model_name in all_models_to_run:
            print(f"\n--- Benchmarking Model: {model_name} on Dataset: {dataset_name}{graph_variant_suffix} ---")
            try:
                if model_name == "ProtGramDirectGCN":
                    if pgd_train_data is None:
                        print("Skipping ProtGramDirectGCN due to pre-processing error.")
                        dataset_results.append({'dataset': f"{dataset_name}{graph_variant_suffix}", 'model': model_name, 'error': "Pre-processing failed"})
                        continue
                    layer_dims = [num_features] + self.config.GCN_HIDDEN_LAYER_DIMS + [num_classes]
                    model_instance = ProtGramDirectGCN(
                        layer_dims=layer_dims, num_graph_nodes=train_data.num_nodes,
                        task_num_output_classes=num_classes, n_gram_len=0, one_gram_dim=0, max_pe_len=0,
                        dropout=self.config.GCN_DROPOUT_RATE, use_vector_coeffs=self.config.GCN_USE_VECTOR_COEFFS
                    ).to(self.device)
                    train_d, val_d, test_d = pgd_train_data, pgd_val_data, pgd_test_data
                else:
                    m_config = model_zoo_config[model_name]
                    model_instance = m_config["class"](in_channels=num_features, out_channels=num_classes, **m_config["params"]).to(self.device)
                    train_d, val_d, test_d = train_data, val_data, test_data

                if self.config.DEBUG_VERBOSE: print(f"  Model Architecture:\n{model_instance}")
                optimizer = torch.optim.Adam(model_instance.parameters(), lr=self.config.EVAL_LEARNING_RATE, weight_decay=5e-4)

                val_metric, test_metric, history_df, metric_name_used = self.train_and_evaluate(
                    model_name, model_instance, f"{dataset_name}{graph_variant_suffix}",
                    train_d, val_d, test_d,
                    optimizer, num_epochs=self.config.EVAL_EPOCHS, task_type=task_type,
                    num_classes=num_classes
                )

                result_entry = {
                    'dataset': f"{dataset_name}{graph_variant_suffix}", 'model': model_name,
                    f'best_val_{metric_name_used.lower().replace(" ", "_")}': val_metric,
                    f'test_{metric_name_used.lower().replace(" ", "_")}': test_metric
                }
                dataset_results.append(result_entry)

                self._save_node_embeddings(model_instance, test_d, model_name, dataset_name, graph_variant_suffix)

                reports_dir = str(self.config.BENCHMARKING_RESULTS_DIR / f"{dataset_name}{graph_variant_suffix}")
                os.makedirs(reports_dir, exist_ok=True)
                history_path = os.path.join(reports_dir, f'benchmark_{model_name}_history.csv')
                history_df.to_csv(history_path, index=False)
                print(f"  Saved {model_name} training history to {history_path}")

            except Exception as e:
                print(f"ERROR during training/evaluation of {model_name} on {dataset_name}{graph_variant_suffix}: {e}")
                import traceback
                traceback.print_exc()
                dataset_results.append({'dataset': f"{dataset_name}{graph_variant_suffix}", 'model': model_name, 'error': str(e)})

        return dataset_results

    def run(self):
        self.set_seeds(self.config.RANDOM_STATE)
        DataUtils.print_header("PIPELINE: GNN BENCHMARKER")
        base_dataset_path = str(self.config.BASE_DATA_DIR / "standard_datasets_pyg")
        os.makedirs(base_dataset_path, exist_ok=True)
        print(f"Standard PyG datasets will be stored in/loaded from: {base_dataset_path}")

        all_results_summary = []

        for dataset_name in self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            train_data_orig, val_data_orig, test_data_orig, task_type_orig = self._get_dataset(dataset_name, base_dataset_path, make_undirected=False)
            if train_data_orig is None:
                print(f"Skipping dataset {dataset_name} (original) due to loading error.")
                continue

            num_features = train_data_orig.x.shape[1]
            num_classes = train_data_orig.y.max().item() + 1 if train_data_orig.y is not None and train_data_orig.y.numel() > 0 else 1

            model_zoo_cfg = {
                "GCN": {"class": GCN, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
                "GAT": {"class": GAT, "params": {"hidden_channels": 32, "heads": 8, "num_layers": 2, "dropout_rate": 0.6}},
                "GraphSAGE": {"class": GraphSAGE, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
                "GIN": {"class": GIN, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
                "ChebNet": {"class": ChebNet, "params": {"hidden_channels": 256, "K": 3, "num_layers": 2, "dropout_rate": 0.5}},
                "RGCN_SR": {"class": RGCN, "params": {"hidden_channels": 256, "num_relations": 1, "num_layers": 2, "dropout_rate": 0.5}},
                "TongDiGCN": {"class": TongDiGCN, "params": {"hidden_channels": 128, "num_layers": 2, "dropout_rate": 0.5}},
            }

            results_orig = self.run_on_dataset_variant(dataset_name, model_zoo_cfg,
                                                       train_data_orig, val_data_orig, test_data_orig,
                                                       task_type_orig, num_features, num_classes, "_Original")
            all_results_summary.extend(results_orig)

            if self.config.BENCHMARK_TEST_ON_UNDIRECTED:
                train_data_undir, val_data_undir, test_data_undir, task_type_undir = self._get_dataset(
                    dataset_name, base_dataset_path, make_undirected=True)
                if train_data_undir is not None:
                    results_undir = self.run_on_dataset_variant(dataset_name, model_zoo_cfg,
                                                                train_data_undir, val_data_undir, test_data_undir,
                                                                task_type_undir, num_features, num_classes, "_Undirected")
                    all_results_summary.extend(results_undir)

            current_dataset_summary = [res for res in all_results_summary if res['dataset'].startswith(dataset_name)]
            if current_dataset_summary:
                dataset_summary_df = pd.DataFrame(current_dataset_summary)
                summary_path = os.path.join(str(self.config.BENCHMARKING_RESULTS_DIR), f"benchmark_summary_{dataset_name}.csv")
                DataUtils.save_dataframe_to_csv(dataset_summary_df, summary_path)
                print(f"\nSummary for {dataset_name} saved to {summary_path}")
                print(dataset_summary_df)

        if all_results_summary:
            final_summary_df = pd.DataFrame(all_results_summary)
            final_summary_path = os.path.join(str(self.config.BENCHMARKING_RESULTS_DIR), "gnn_benchmark_FULL_SUMMARY.csv")
            DataUtils.save_dataframe_to_csv(final_summary_df, final_summary_path)
            print(f"\nFull GNN benchmarking summary saved to {final_summary_path}")
            print("\nFull Summary Table:")
            print(final_summary_df.to_string())

        DataUtils.print_header("GNN Benchmarking PIPELINE FINISHED")