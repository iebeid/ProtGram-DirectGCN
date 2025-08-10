# ==============================================================================
# MODULE: benchmarkers/nes.py
# PURPOSE: Handles benchmarking of Network Embedding models like Node2Vec.
# VERSION: 3.0 (Fixed WebKB mask handling and updated metrics reporting)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import traceback
from typing import Dict, Any, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WebKB, Actor, KarateClub
from torch_geometric.nn import Node2Vec

from configuration.config import Config
from source.utils.data import DataUtils


class SimpleMLP(torch.nn.Module):
    """A simple PyTorch MLP for node classification on embeddings."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        if num_layers == 1:
            self.layers.append(torch.nn.Linear(in_channels, out_channels))
        else:
            self.layers.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x


class NetworkEmbeddingBenchmarker:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = config.RESULTS_BENCHMARKING_DIR
        self.dataset_root = str(config.DATA_STANDARD_DATASETS_DIR)
        print("\n" + "=" * 80)
        DataUtils.print_header("Network Embedding Benchmarker Initialized")
        print(f"  Device: {self.device}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Dataset root: {self.dataset_root}")
        print("=" * 80)
        # --- NEW: Set seeds for reproducibility of Node2Vec and MLP training ---
        DataUtils.set_seeds(self.config.RANDOM_STATE)

    def _get_dataset(self, name: str) -> Any:
        """Loads a standard PyG dataset."""
        path = self.dataset_root
        try:
            if name in ['Cora', 'CiteSeer', 'PubMed']:
                return Planetoid(root=path, name=name)
            elif name in ['Cornell', 'Texas', 'Wisconsin']:
                return WebKB(root=path, name=name)
            elif name == 'Actor':
                return Actor(root=path)
            elif name == 'KarateClub':
                return KarateClub()
            else:
                print(f"  Dataset '{name}' not recognized by this loader.")
                return None
        except Exception as e:
            print(f"  Error loading dataset '{name}': {e}")
            return None

    def _get_1d_mask(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """Helper to handle masks from datasets that may have multiple splits (e.g., WebKB)."""
        if mask_tensor.dim() > 1:
            # WebKB datasets have a [num_nodes, 10] mask for 10 splits. We use the first one.
            return mask_tensor[:, 0].bool()
        return mask_tensor.bool()

    def _train_and_evaluate_mlp(self, embeddings: torch.Tensor, data: Data) -> Dict[str, float]:
        """Trains and evaluates a simple MLP on the generated embeddings."""
        num_classes = int(data.y.max().item()) + 1
        mlp = SimpleMLP(
            in_channels=embeddings.shape[1],
            hidden_channels=128,
            out_channels=num_classes,
            dropout=0.5
        ).to(self.device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)

        # FIX: Use the helper to handle masks that might have multiple splits (e.g., WebKB)
        train_mask = self._get_1d_mask(data.train_mask)
        val_mask = self._get_1d_mask(data.val_mask)
        test_mask = self._get_1d_mask(data.test_mask)

        best_val_acc = -1
        test_acc_at_best_val = -1
        f1_at_best_val = -1
        precision_at_best_val = -1
        recall_at_best_val = -1

        for epoch in range(1, 201):  # A fixed number of epochs for the MLP classifier
            mlp.train()
            optimizer.zero_grad()
            out = mlp(embeddings[train_mask])
            loss = F.cross_entropy(out, data.y[train_mask])
            loss.backward()
            optimizer.step()

            mlp.eval()
            with torch.no_grad():
                pred = mlp(embeddings).argmax(dim=1)
                val_correct = (pred[val_mask] == data.y[val_mask]).sum()
                val_acc = int(val_correct) / int(val_mask.sum()) if val_mask.sum() > 0 else 0.0
                test_correct = (pred[test_mask] == data.y[test_mask]).sum()
                test_acc = int(test_correct) / int(test_mask.sum()) if test_mask.sum() > 0 else 0.0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val = test_acc
                # --- NEW: Capture all test metrics at the best validation epoch ---
                y_true_test = data.y[test_mask].cpu().numpy()
                y_pred_test = pred[test_mask].cpu().numpy()
                f1_at_best_val = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)
                precision_at_best_val = precision_score(y_true_test, y_pred_test, average='macro', zero_division=0)
                recall_at_best_val = recall_score(y_true_test, y_pred_test, average='macro', zero_division=0)

        metrics = {
            'Accuracy': test_acc_at_best_val,
            'F1-Score (Macro)': f1_at_best_val,
            'Precision (Macro)': precision_at_best_val,
            'Recall (Macro)': recall_at_best_val,
            'best_val_accuracy': best_val_acc
        }
        return metrics

    def _run_on_dataset(self, dataset: Any, dataset_name: str, model_name: str) -> Dict:
        """Runs a single NE model on a single dataset."""
        data = dataset[0].to(self.device)
        print(f"--- Benchmarking Model: {model_name} on Dataset: {dataset_name} ---")

        if not all(hasattr(data, mask) and getattr(data, mask) is not None and getattr(data, mask).any() for mask in ['train_mask', 'val_mask', 'test_mask']):
            print(f"  - No predefined splits found for {dataset_name}. Creating random splits.")
            num_nodes = data.num_nodes
            rng = np.random.default_rng(self.config.RANDOM_STATE) # Use a seeded generator
            indices = rng.permutation(num_nodes)
            train_size = int(num_nodes * 0.1)
            val_size = int(num_nodes * 0.1)
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
            data.train_mask[indices[:train_size]] = True
            data.val_mask[indices[train_size:train_size + val_size]] = True
            data.test_mask[indices[train_size + val_size:]] = True
            print(f"  Generated custom seeded split for {dataset_name}. Train: {train_size}, Val: {val_size}, Test: {int(data.test_mask.sum())}")

        # 1. Train Node2Vec to get embeddings
        node2vec_model = Node2Vec(
            data.edge_index,
            embedding_dim=self.config.BENCHMARK_NE_EMBEDDING_DIM,
            walk_length=self.config.BENCHMARK_NE_WALK_LENGTH,
            context_size=self.config.BENCHMARK_NE_CONTEXT_SIZE,
            walks_per_node=10,
            num_negative_samples=1,
            p=1, q=1,
            sparse=True,
        ).to(self.device)

        # --- DEFINITIVE FIX for hidden prompt and reproducibility: Set num_workers=0 ---
        # Using multiple worker processes (num_workers > 0) can cause two issues:
        # 1. The main script hangs before an input() prompt, waiting for background processes.
        # 2. It introduces non-determinism unless a specific `worker_init_fn` is used.
        # Setting num_workers=0 forces data loading to happen in the main thread, resolving both.
        loader = node2vec_model.loader(batch_size=128, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(list(node2vec_model.parameters()), lr=0.01)

        for _ in range(self.config.BENCHMARK_NE_EPOCHS):
            node2vec_model.train()
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = node2vec_model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()

        # 2. Get the final embeddings
        with torch.no_grad():
            node2vec_model.eval()
            embeddings = node2vec_model()

        # 3. Train and evaluate an MLP on the embeddings
        metrics = self._train_and_evaluate_mlp(embeddings, data)
        print(f"  ✅ Best Val Acc: {metrics.get('best_val_accuracy', -1):.4f}, Test Accuracy for {model_name} on {dataset_name}: {metrics.get('Accuracy', -1):.4f}")

        result_row = {"dataset": dataset_name, "model": model_name, "error": None}
        result_row.update(metrics)

        # --- FIX: Explicitly delete large torch objects to release memory and resources ---
        # This helps prevent the loader from hanging in the background before the main script prompts for user input.
        del node2vec_model, loader, optimizer, embeddings

        return result_row

    def run(self) -> pd.DataFrame:
        """Main execution function for the benchmarker."""
        DataUtils.print_header("PIPELINE: Network Embedding BENCHMARKER")
        all_results = []

        for dataset_name in self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            dataset = self._get_dataset(dataset_name)
            if dataset is None: continue
            print(f"\nLoaded dataset: {dataset_name}. Nodes: {dataset[0].num_nodes}, Edges: {dataset[0].num_edges}")

            for model_name in self.config.BENCHMARK_NE_MODELS_TO_RUN:
                try:
                    with mlflow.start_run(run_name=f"{model_name}_on_{dataset_name}", nested=True) as run:
                        mlflow.set_tag("model_name", model_name)
                        mlflow.set_tag("dataset_name", dataset_name)
                        result = self._run_on_dataset(dataset, dataset_name, model_name)
                        all_results.append(result)
                        mlflow.log_metrics({
                            "best_val_accuracy": result.get('best_val_accuracy', 0.0),
                            "test_accuracy": result.get('Accuracy', 0.0)
                        })
                except Exception as e:
                    print(f"ERROR during benchmarking of {model_name} on {dataset_name}: {e}")
                    traceback.print_exc()
                    all_results.append({"dataset": dataset_name, "model": model_name, "error": str(e)})

        # --- FIX: Force garbage collection after the benchmark loop ---
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        summary_df = pd.DataFrame(all_results)
        # The full summary is now handled by main.py
        DataUtils.print_header("Network Embedding BENCHMARKER FINISHED")
        return summary_df if all_results else pd.DataFrame()