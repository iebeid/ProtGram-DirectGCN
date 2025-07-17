# ==============================================================================
# MODULE: benchmarkers/gnns.py
# PURPOSE: Handles benchmarking of various GNN models on standard datasets.
# VERSION: 1.1 (Corrected dataset root path and RGCN forward pass)
# AUTHOR: Your Name (Assembled by Coding Partner)
# ==============================================================================

import os
import time
from typing import Dict, Optional, List
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, GNNBenchmarkDataset, ZINC, TUDataset, QM9
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
        # FIX: Use the correct dataset directory from the config
        self.dataset_root = str(config.DATA_STANDARD_DATASETS_DIR)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.embedding_dir, exist_ok=True)

        print(f"GNNBenchmarker initialized. Using device: {self.device}")
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
                # For WebKB datasets, we need to handle the transform slightly differently
                # as they don't natively come with a directed option to undo.
                return WebKB(root=path, name=name)
            elif name == 'Actor':
                return Actor(root=path)
            elif name == 'KarateClub':
                from torch_geometric.datasets import KarateClub
                return KarateClub() # This one doesn't take a root
            else:
                print(f"  Dataset '{name}' not recognized by this loader.")
                return None
        except Exception as e:
            print(f"  Error loading dataset '{name}': {e}")
            return None

    def _get_model(self, name: str, data, num_relations: int = 1):
        model_params = {
            "GCN": {"class": GCN, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
            "GAT": {"class": GAT, "params": {"hidden_channels": 32, "heads": 8, "num_layers": 2, "dropout_rate": 0.6}},
            "GraphSAGE": {"class": GraphSAGE, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
            "GIN": {"class": GIN, "params": {"hidden_channels": 256, "num_layers": 2, "dropout_rate": 0.5}},
            "ChebNet": {"class": ChebNet, "params": {"hidden_channels": 256, "K": 3, "num_layers": 2, "dropout_rate": 0.5}},
            "RGCN_SR": {"class": RGCN, "params": {"hidden_channels": 256, "num_relations": num_relations, "num_layers": 2, "dropout_rate": 0.5}},
            "TongDiGCN": {"class": TongDiGCN, "params": {"hidden_dim": 128}},
            "ProtGramDirectGCN": {"class": ProtGramDirectGCN, "params": {"layer_dims": [data.num_features, 256, 128, 64, data.num_classes], "num_graph_nodes": data.num_nodes, "n_gram_len": 0, "one_gram_dim": 0, "max_pe_len": 0, "dropout": 0.5, "use_vector_coeffs": False}}
        }
        model_info = model_params.get(name)
        if not model_info:
            raise ValueError(f"Model {name} not found in GNNBenchmarker.")

        params = model_info['params']
        # Standardize parameter names for model constructors
        if name not in ["ProtGramDirectGCN", "TongDiGCN"]:
            params['in_channels'] = data.num_features
            params['out_channels'] = data.num_classes

        return model_info['class'](**params)


    def _preprocess_for_directgcn(self, data):
        """Prepares standard PyG data for ProtGramDirectGCN's specific input format."""
        print(f"--- Pre-processing data for ProtGramDirectGCN on {data.name} ---")
        # 1. Create undirected normalized matrix (used for the structural path)
        edge_index_undir, _ = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        row, col = edge_index_undir
        deg = torch.bincount(col, minlength=data.num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight_undir = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        data.edge_index_undirected_norm = edge_index_undir
        data.edge_weight_undirected_norm = edge_weight_undir

        # 2. Create directional matrices (in/out)
        # For standard datasets, we can treat the original edge_index as the "out" edges
        # and its transpose as the "in" edges. They won't have inherent weights.
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

        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            out = model(train_data)

            # The output 'out' might be a tuple from ProtGramDirectGCN
            if isinstance(out, tuple):
                out = out[0]

            loss = F.cross_entropy(out[train_data.train_mask], train_data.y[train_data.train_mask])
            loss.backward()
            optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                out = model(train_data)
                if isinstance(out, tuple): out = out[0]

                pred = out.argmax(dim=1)
                val_correct = pred[val_data.val_mask] == val_data.y[val_data.val_mask]
                val_acc = int(val_correct.sum()) / int(val_data.val_mask.sum())

                test_correct = pred[test_data.test_mask] == test_data.y[test_data.test_mask]
                test_acc = int(test_correct.sum()) / int(test_data.test_mask.sum())

                val_loss = F.cross_entropy(out[val_data.val_mask], val_data.y[val_data.val_mask])

            history['epoch'].append(epoch)
            history['loss'].append(loss.item())
            history['val_loss'].append(val_loss.item())
            history['val_metric'].append(val_acc)
            history['test_metric'].append(test_acc)

            if val_acc > best_val_metric:
                best_val_metric = val_acc
                corresponding_test_metric = test_acc

            if self.config.DEBUG_VERBOSE and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1):
                 print(f"    Epoch {epoch:03d}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}")

        print(f"  Finished training for {model.__class__.__name__} on {train_data.name}.")
        print(f"  Best Val Accuracy: {best_val_metric:.4f}, Corresponding Test Accuracy: {corresponding_test_metric:.4f}")

        # Extract embeddings if configured
        if self.config.BENCHMARK_SAVE_EMBEDDINGS:
            print(f"    Extracting embeddings for {model.__class__.__name__} on {train_data.name}...")
            with torch.no_grad():
                model.eval()
                # BaseGNN models store embeddings in .embedding_output after forward pass
                if hasattr(model, 'get_embeddings'):
                    full_embeddings = model.get_embeddings(train_data.to(self.device))
                    if isinstance(full_embeddings, tuple): full_embeddings = full_embeddings[1]
                else:
                    # Fallback for models without get_embeddings
                    output = model(train_data.to(self.device))
                    full_embeddings = output[0] if isinstance(output, tuple) else output

            if full_embeddings is not None:
                embeddings_np = full_embeddings.cpu().numpy()
                emb_dim = embeddings_np.shape[1]
                emb_dict = {str(i): embeddings_np[i] for i in range(embeddings_np.shape[0])}

                save_path_emb_dir = self.embedding_dir / train_data.name
                os.makedirs(save_path_emb_dir, exist_ok=True)

                if self.config.BENCHMARK_APPLY_PCA_TO_EMBEDDINGS and emb_dim > self.config.BENCHMARK_PCA_TARGET_DIM:
                    print(f"      Applying PCA to {model.__class__.__name__} embeddings (target dim: {self.config.BENCHMARK_PCA_TARGET_DIM})...")
                    pca_embeds = EmbeddingProcessor.apply_pca(emb_dict, self.config.BENCHMARK_PCA_TARGET_DIM, self.config.RANDOM_STATE)
                    if pca_embeds:
                        h5_path = save_path_emb_dir / f"{model.__class__.__name__}_embeddings_pca{self.config.BENCHMARK_PCA_TARGET_DIM}.h5"
                        with h5py.File(h5_path, 'w') as hf:
                            for k, v in pca_embeds.items(): hf.create_dataset(k, data=v)
                        print(f"      Saved {model.__class__.__name__} embeddings for {train_data.name} to {h5_path}")
                else:
                    h5_path = save_path_emb_dir / f"{model.__class__.__name__}_embeddings_dim{emb_dim}.h5"
                    with h5py.File(h5_path, 'w') as hf:
                        for k, v in emb_dict.items(): hf.create_dataset(k, data=v)
                    print(f"      Saved {model.__class__.__name__} embeddings for {train_data.name} to {h5_path}")

        return best_val_metric, corresponding_test_metric, pd.DataFrame(history), metric_name

    def run_on_dataset_variant(self, dataset, variant_name: str):
        print(f"\n" + "=" * 50)
        print(f"### Benchmarking on Dataset: {variant_name} ###")
        print("=" * 50 + "\n")

        # Use the first data object
        data = dataset[0]
        data.name = variant_name

        # Generate custom split if no masks exist
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            print(f"  Generating custom seeded split for {dataset.name}.")
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
            print(f"  Using existing standard masks for {dataset.name}.")

        print(f"  {dataset.name} loaded: Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_features}, Classes: {dataset.num_classes}")

        # Pre-process a copy for ProtGramDirectGCN if it's in the list
        models_to_run = self.config.GNN_MODELS_TO_RUN if hasattr(self.config, 'GNN_MODELS_TO_RUN') else ["GCN", "GAT", "GraphSAGE", "GIN", "ChebNet", "RGCN_SR", "TongDiGCN", "ProtGramDirectGCN"]
        data_for_protgram = self._preprocess_for_directgcn(data.clone()) if "ProtGramDirectGCN" in models_to_run else None

        results = []
        for model_name in models_to_run:
            print(f"\n--- Benchmarking Model: {model_name} on Dataset: {variant_name} ---")
            try:
                data_to_use = data_for_protgram if model_name == 'ProtGramDirectGCN' else data
                if data_to_use is None: continue

                model = self._get_model(model_name, data_to_use)
                print("  Model Architecture:")
                print(model)

                epochs = self.config.EVAL_EPOCHS
                print(f"  Training {model_name} on {variant_name} using device: {self.device} for {epochs} epochs.")
                print(f"  Using Loss: cross_entropy, Metric: Accuracy")

                val_metric, test_metric, history_df, metric_name_used = self.train_and_evaluate(
                    model=model,
                    train_data=data_to_use,
                    val_data=data_to_use,
                    test_data=data_to_use,
                    loss_fn_name='cross_entropy',
                    metric_name='accuracy',
                    epochs=epochs
                )

                results.append({"dataset": variant_name, "model": model_name,
                                "best_val_accuracy": val_metric, "test_accuracy": test_metric, "error": None})

                history_path = self.output_dir / variant_name
                os.makedirs(history_path, exist_ok=True)
                history_df.to_csv(history_path / f"benchmark_{model_name}_history.csv", index=False)
                print(f"  Saved {model_name} training history to {history_path / f'benchmark_{model_name}_history.csv'}")

            except Exception as e:
                print(f"ERROR during training/evaluation of {model_name} on {variant_name}: {e}")
                import traceback
                traceback.print_exc()
                results.append({"dataset": variant_name, "model": model_name,
                                "best_val_accuracy": None, "test_accuracy": None, "error": str(e)})
        return results


    def run(self):
        DataUtils.print_header("PIPELINE: GNN BENCHMARKER")
        all_results = []

        print(f"Standard PyG datasets will be stored in/loaded from: {self.dataset_root}")

        for dataset_name in self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            # 1. Run on the original (potentially directed) graph
            print(f"\n  Attempting to load dataset: {dataset_name} (root: {self.dataset_root}, undirected_requested: False)...")
            dataset = self._get_dataset(dataset_name, undirected=False)
            if dataset:
                all_results.extend(self.run_on_dataset_variant(dataset, f"{dataset_name}_Original"))

            # 2. Run on the undirected version of the graph
            if self.config.BENCHMARK_TEST_ON_UNDIRECTED:
                print(f"\n  Attempting to load dataset: {dataset_name} (root: {self.dataset_root}, undirected_requested: True)...")
                dataset_undirected = self._get_dataset(dataset_name, undirected=True)
                if dataset_undirected:
                    all_results.extend(self.run_on_dataset_variant(dataset_undirected, f"{dataset_name}_Undirected"))

            if all_results:
                summary_df = pd.DataFrame([r for r in all_results if dataset_name in r['dataset']])
                summary_path = self.output_dir / f"benchmark_summary_{dataset_name}.csv"
                DataUtils.save_dataframe_to_csv(summary_df, str(summary_path))
                print(f"\nSummary for {dataset_name} saved to {summary_path}")
                print(summary_df)

        if all_results:
            full_summary_df = pd.DataFrame(all_results)
            full_summary_path = self.output_dir / "gnn_benchmark_FULL_SUMMARY.csv"
            DataUtils.save_dataframe_to_csv(full_summary_df, str(full_summary_path))
            print(f"\nFull GNN benchmarking summary saved to {full_summary_path}")
            print("\nFull Summary Table:")
            print(full_summary_df.to_string())

        DataUtils.print_header("GNN Benchmarking PIPELINE FINISHED")