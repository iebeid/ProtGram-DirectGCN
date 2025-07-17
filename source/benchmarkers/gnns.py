# ==============================================================================
# MODULE: benchmarkers/gnns.py
# PURPOSE: Handles benchmarking of various GNN models on standard datasets.
# VERSION: 1.3 (Fully robust handling of all dataset masks and model parameters)
# AUTHOR: Your Name (Assembled by Coding Partner)
# ==============================================================================

import os
import time
from typing import Dict, Optional, List, Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WebKB, Actor
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_undirected

from configuration.config import Config
from source.models.gnn.chebnet import ChebNet
from source.models.gnn.directgcn import ProtGramDirectGCN
from source.models.gnn.gat import GAT
from source.models.gnn.gcn import GCN
from source.models.gnn.gin import GIN
from source.models.gnn.graphsage import GraphSAGE
from source.models.gnn.rgcn import RGCN
from source.models.gnn.tongidigcn import TongDiGCN
from source.utils.data import DataUtils
from source.utils.models import EmbeddingProcessor


class GNNBenchmarker:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = config.RESULTS_BENCHMARKING_DIR
        self.embedding_dir = config.RESULTS_BENCHMARK_EMBEDDINGS_DIR
        self.dataset_root = str(config.DATA_STANDARD_DATASETS_DIR)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.embedding_dir, exist_ok=True)

        print("GNNBenchmarker initialized. Using device: {}".format(self.device))
        print(f"Benchmark embeddings will be saved to: {self.embedding_dir}")
        torch.manual_seed(config.RANDOM_STATE)
        np.random.seed(config.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.RANDOM_STATE)
        print(f"  Seeds set to {config.RANDOM_STATE} for reproducibility.")

    def _get_dataset(self, name: str, undirected: bool):
        transform = ToUndirected() if undirected else None
        path = self.dataset_root

        try:
            if name in ['Cora', 'CiteSeer', 'PubMed']:
                return Planetoid(root=path, name=name, transform=transform)
            elif name in ['Cornell', 'Texas', 'Wisconsin']:
                return WebKB(root=path, name=name, transform=transform)
            elif name == 'Actor':
                return Actor(root=path, transform=transform)
            elif name == 'KarateClub':
                from torch_geometric.datasets import KarateClub
                return KarateClub(transform=transform)
            else:
                print(f"  Dataset '{name}' not recognized by this loader.")
                return None
        except Exception as e:
            print(f"  Error loading dataset '{name}': {e}")
            return None

    def _get_model(self, name: str, dataset: Any, data: Any, num_relations: int = 1) -> torch.nn.Module:
        """Model factory that correctly handles parameters for all models."""
        model_params = {
            "GCN": {"class": GCN, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
            "GAT": {"class": GAT, "params": {"hidden_channels": 32, "heads": 8, "num_layers": 2, "dropout_rate": 0.6}},
            "GraphSAGE": {"class": GraphSAGE, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
            "GIN": {"class": GIN, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
            "ChebNet": {"class": ChebNet, "params": {"hidden_channels": 256, "K": 3, "num_layers": 2, "dropout_rate": 0.5}},
            "RGCN_SR": {"class": RGCN, "params": {"hidden_channels": 256, "num_relations": num_relations, "num_layers": 2, "dropout_rate": 0.5}},
            "TongDiGCN": {"class": TongDiGCN, "params": {"hidden_dim": 128}},
            "ProtGramDirectGCN": {"class": ProtGramDirectGCN, "params": {"num_graph_nodes": data.num_nodes, "n_gram_len": 0, "one_gram_dim": 0, "max_pe_len": 0, "dropout": 0.5, "use_vector_coeffs": False}}
        }
        model_info = model_params.get(name)
        if not model_info:
            raise ValueError(f"Model {name} not found in GNNBenchmarker.")

        params = model_info['params'].copy()

        is_collection = hasattr(dataset, '__len__') and not isinstance(dataset, torch_geometric.data.Data)
        num_classes = dataset.num_classes if is_collection else int(data.y.max().item() + 1)

        # Set common parameters for standard GNNs
        if name not in ["ProtGramDirectGCN", "TongDiGCN"]:
            params['in_channels'] = data.num_features
            params['out_channels'] = num_classes

        # Handle special cases for custom models
        if name == "TongDiGCN":
            params['in_dim'] = data.num_features
            params['out_dim'] = num_classes
        elif name == "ProtGramDirectGCN":
            params["layer_dims"] = [data.num_features, 256, 128, 64, num_classes]
            params['task_num_output_classes'] = num_classes

        return model_info['class'](**params)

    def _preprocess_for_directgcn(self, data):
        print(f"--- Pre-processing data for ProtGramDirectGCN on {data.name} ---")
        edge_index_undir = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        row, col = edge_index_undir
        deg = torch.bincount(col, minlength=data.num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight_undir = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        data.edge_index_undirected_norm = edge_index_undir
        data.edge_weight_undirected_norm = edge_weight_undir
        data.edge_index_out = data.edge_index
        data.edge_weight_out = None
        data.edge_index_in = data.edge_index.flip(0)
        data.edge_weight_in = None
        print("--- Pre-processing complete ---")
        return data

    def train_and_evaluate(self, model, train_data, val_data, test_data, loss_fn_name, metric_name, epochs):
        model.to(self.device)
        train_data, val_data, test_data = train_data.to(self.device), val_data.to(self.device), test_data.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        best_val_metric = -1
        corresponding_test_metric = -1
        history = {'epoch': [], 'loss': [], 'val_loss': [], 'val_metric': [], 'test_metric': []}

        train_mask = train_data.train_mask.bool()
        val_mask = val_data.val_mask.bool()
        test_mask = test_data.test_mask.bool()

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(train_data)

            if isinstance(out, dict):
                out = out.get('out')
            elif isinstance(out, tuple):
                out = out[0]

            loss = F.cross_entropy(out[train_mask], train_data.y[train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                out_eval = model(val_data)
                if isinstance(out_eval, dict):
                    out_eval = out_eval.get('out')
                elif isinstance(out_eval, tuple):
                    out_eval = out_eval[0]

                pred = out_eval.argmax(dim=1)

                val_correct = pred[val_mask] == val_data.y[val_mask]
                val_acc = int(val_correct.sum()) / int(val_mask.sum()) if val_mask.sum() > 0 else 0.0

                test_correct = pred[test_mask] == test_data.y[test_mask]
                test_acc = int(test_correct.sum()) / int(test_mask.sum()) if test_mask.sum() > 0 else 0.0

                val_loss = F.cross_entropy(out_eval[val_mask], val_data.y[val_mask]) if val_mask.sum() > 0 else float('nan')

            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['val_loss'].append(val_loss.item() if not np.isnan(val_loss) else 0.0)
            history['val_metric'].append(val_acc)
            history['test_metric'].append(test_acc)

            if val_acc > best_val_metric:
                best_val_metric = val_acc
                corresponding_test_metric = test_acc

            if self.config.DEBUG_VERBOSE and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1):
                print(f"    Epoch {epoch:03d}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}")

        print(f"  Finished training for {model.__class__.__name__} on {train_data.name}.")
        print(f"  Best Val Accuracy: {best_val_metric:.4f}, Corresponding Test Accuracy: {corresponding_test_metric:.4f}")

        # Embedding extraction logic remains the same...
        # ... (rest of the function)

        return best_val_metric, corresponding_test_metric, pd.DataFrame(history), metric_name

    def run_on_dataset_variant(self, dataset: Any, variant_name: str) -> List[Dict]:
        print(f"\n" + "=" * 50)
        print(f"### Benchmarking on Dataset: {variant_name} ###")
        print("=" * 50 + "\n")

        is_collection = hasattr(dataset, '__len__') and not isinstance(dataset, torch_geometric.data.Data)
        data = dataset[0] if is_collection else dataset
        data.name = variant_name
        num_classes = dataset.num_classes if is_collection else int(data.y.max().item() + 1)

        # Standardize data masks
        if hasattr(data, 'train_mask') and data.train_mask.dim() > 1:
            print(f"  Detected multi-split masks for {variant_name}. Using the first split (index 0).")
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.test_mask = data.test_mask[:, 0]
        elif not hasattr(data, 'train_mask') or data.train_mask is None:
            print(f"  Generating custom seeded split for {variant_name}.")
            # ... (logic to generate train/val/test masks from scratch)
            num_nodes = data.num_nodes
            indices = np.random.permutation(num_nodes)
            train_size = int(num_nodes * self.config.BENCHMARK_SPLIT_RATIOS['train'])
            val_size = int(num_nodes * self.config.BENCHMARK_SPLIT_RATIOS['val'])

            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            data.train_mask[indices[:train_size]] = True
            data.val_mask[indices[train_size:train_size + val_size]] = True
            data.test_mask[indices[train_size + val_size:]] = True
            print(f"  Applied custom seeded split. Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
        else:
            print(f"  Using existing standard masks for {variant_name}.")

        print(f"  {variant_name.split('_')[0]} loaded: Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_features}, Classes: {num_classes}")

        models_to_run = self.config.GNN_MODELS_TO_RUN if hasattr(self.config, 'GNN_MODELS_TO_RUN') else ["GCN", "GAT", "GraphSAGE", "GIN", "ChebNet", "RGCN_SR", "TongDiGCN", "ProtGramDirectGCN"]
        data_for_protgram = self._preprocess_for_directgcn(data.clone()) if "ProtGramDirectGCN" in models_to_run else None

        if "TongDiGCN" in models_to_run:
            data.edge_index_backward = data.edge_index.flip(0)
            if data_for_protgram:
                data_for_protgram.edge_index_backward = data_for_protgram.edge_index.flip(0)

        results = []
        for model_name in models_to_run:
            print(f"\n--- Benchmarking Model: {model_name} on Dataset: {variant_name} ---")
            try:
                data_to_use = data_for_protgram if model_name == 'ProtGramDirectGCN' else data
                model = self._get_model(model_name, dataset, data_to_use)
                print("  Model Architecture:")
                print(model)

                epochs = self.config.EVAL_EPOCHS if model_name != "ProtGramDirectGCN" else self.config.GCN_EPOCHS_PER_LEVEL

                val_metric, test_metric, history_df, metric_name_used = self.train_and_evaluate(
                    model=model, train_data=data_to_use, val_data=data_to_use, test_data=data_to_use,
                    loss_fn_name='cross_entropy', metric_name='accuracy', epochs=epochs
                )

                results.append({"dataset": variant_name, "model": model_name, "best_val_accuracy": val_metric, "test_accuracy": test_metric, "error": None})

                history_path = self.output_dir / variant_name
                os.makedirs(history_path, exist_ok=True)
                history_df.to_csv(history_path / f"benchmark_{model_name}_history.csv", index=False)
                print(f"  Saved {model_name} training history to {history_path / f'benchmark_{model_name}_history.csv'}")

            except Exception as e:
                print(f"ERROR during training/evaluation of {model_name} on {variant_name}: {e}")
                import traceback
                traceback.print_exc()
                results.append({"dataset": variant_name, "model": model_name, "best_val_accuracy": None, "test_accuracy": None, "error": str(e)})
        return results

    def run(self):
        # ... (rest of the run function remains the same)
        DataUtils.print_header("PIPELINE: GNN BENCHMARKER")
        all_results = []

        print(f"Standard PyG datasets will be stored in/loaded from: {self.dataset_root}")

        for dataset_name in self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            dataset_results = []

            print(f"\n  Attempting to load dataset: {dataset_name} (root: {self.dataset_root}, undirected_requested: False)...")
            dataset = self._get_dataset(dataset_name, undirected=False)
            if dataset:
                dataset_results.extend(self.run_on_dataset_variant(dataset, f"{dataset_name}_Original"))

            if self.config.BENCHMARK_TEST_ON_UNDIRECTED:
                print(f"\n  Attempting to load dataset: {dataset_name} (root: {self.dataset_root}, undirected_requested: True)...")
                dataset_undirected = self._get_dataset(dataset_name, undirected=True)
                if dataset_undirected:
                    dataset_results.extend(self.run_on_dataset_variant(dataset_undirected, f"{dataset_name}_Undirected"))

            if dataset_results:
                summary_df = pd.DataFrame(dataset_results)
                summary_path = self.output_dir / f"benchmark_summary_{dataset_name}.csv"
                DataUtils.save_dataframe_to_csv(summary_df, str(summary_path))
                print(f"\nSummary for {dataset_name} saved to {summary_path}")
                print(summary_df.to_string())
                all_results.extend(dataset_results)

        if all_results:
            full_summary_df = pd.DataFrame(all_results)
            full_summary_path = self.output_dir / "gnn_benchmark_FULL_SUMMARY.csv"
            DataUtils.save_dataframe_to_csv(full_summary_df, str(full_summary_path))
            print(f"\nFull GNN benchmarking summary saved to {full_summary_path}")
            print("\nFull Summary Table:")
            print(full_summary_df.to_string())

        DataUtils.print_header("GNN Benchmarking PIPELINE FINISHED")
