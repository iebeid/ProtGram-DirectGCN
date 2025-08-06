# ==============================================================================
# MODULE: benchmarkers/nes.py
# PURPOSE: Handles benchmarking of traditional network embedding methods.
# VERSION: 3.0 (Integrated MLflow logging and suppressed tokenizer warnings)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import traceback

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import Planetoid, WebKB, KarateClub
from torch_geometric.nn.models import Node2Vec

from configuration.config import Config
from source.utils.data import DataUtils


class NetworkEmbeddingBenchmarker:
    def __init__(self, config: Config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = config.RESULTS_BENCHMARKING_DIR
        self.dataset_root = str(config.DATA_STANDARD_DATASETS_DIR)
        self.models_to_run = [m for m in config.BENCHMARK_NE_MODELS_TO_RUN if m != 'MetaPath2Vec']
        os.makedirs(self.output_dir, exist_ok=True)
        print("\n" + "=" * 80)
        print("### Network Embedding Benchmarker Initialized ###")
        print(f"  Device: {self.device}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Dataset root: {self.dataset_root}")
        print("=" * 80)

    def _get_dataset(self, name: str):
        path = self.dataset_root
        try:
            if name in ['Cora', 'CiteSeer', 'PubMed']:
                return Planetoid(root=path, name=name)
            elif name in ['Cornell', 'Texas', 'Wisconsin']:
                return WebKB(root=path, name=name)
            elif name == 'KarateClub':
                dataset = KarateClub()
                # Manually add the name attribute for consistency
                dataset.name = 'KarateClub'
                return dataset
            else:
                print(f"  Warning: Unknown dataset '{name}' requested.")
                return None
        except Exception as e:
            print(f"  Error loading dataset '{name}': {e}")
            return None

    def _run_model_on_dataset(self, model_name: str, dataset: 'Dataset'):
        data = dataset[0]
        dataset_name = dataset.name
        data = data.to(self.device)

        print(f"--- Benchmarking Model: {model_name} on Dataset: {dataset_name} ---")
        # --- MLFLOW INTEGRATION: Start a nested run for this specific experiment ---
        with mlflow.start_run(run_name=f"{model_name}_on_{dataset_name}", nested=True):
            try:
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("dataset_name", dataset_name)
                mlflow.log_param("embedding_dim", self.config.BENCHMARK_NE_EMBEDDING_DIM)
                mlflow.log_param("epochs", self.config.BENCHMARK_NE_EPOCHS)
                mlflow.log_param("walk_length", self.config.BENCHMARK_NE_WALK_LENGTH)
                mlflow.log_param("context_size", self.config.BENCHMARK_NE_CONTEXT_SIZE)

                if model_name == 'Node2Vec':
                    model = Node2Vec(
                        edge_index=data.edge_index,
                        embedding_dim=self.config.BENCHMARK_NE_EMBEDDING_DIM,
                        walk_length=self.config.BENCHMARK_NE_WALK_LENGTH,
                        context_size=self.config.BENCHMARK_NE_CONTEXT_SIZE,
                        walks_per_node=10,
                        num_negative_samples=1,
                        p=1,
                        q=1,
                        sparse=True,
                    ).to(self.device)
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                num_workers = getattr(self.config, 'GRAPH_BUILDER_WORKERS', 0)
                loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
                optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

                for _ in range(1, self.config.BENCHMARK_NE_EPOCHS + 1):
                    model.train()
                    for pos_rw, neg_rw in loader:
                        optimizer.zero_grad()
                        loss = model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                        loss.backward()
                        optimizer.step()

                model.eval()
                with torch.no_grad():
                    z = model().detach()

                # --- FIX: Robustly check for all three masks before using them ---
                if not all(hasattr(data, mask) and getattr(data, mask) is not None for mask in
                           ['train_mask', 'val_mask', 'test_mask']):
                    print(f"  - No predefined splits found for {dataset_name}. Creating random splits.")
                    num_nodes = data.num_nodes
                    # FIX: Use a seeded random number generator for reproducible splits.
                    rng = np.random.default_rng(self.config.RANDOM_STATE)
                    indices = rng.permutation(num_nodes)
                    train_size = int(num_nodes * 0.1)
                    val_size = int(num_nodes * 0.1)

                    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
                    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
                    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)

                    data.train_mask[indices[:train_size]] = True
                    data.val_mask[indices[train_size:train_size + val_size]] = True
                    data.test_mask[indices[train_size + val_size:]] = True
                    print(
                        f"  Generated custom seeded split for {dataset_name}. Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
                # --- END FIX ---

                # Access the masks *after* they are guaranteed to exist.
                if hasattr(data, 'train_mask') and data.train_mask.dim() > 1:
                    train_mask = data.train_mask[:, 0].bool()
                    test_mask = data.test_mask[:, 0].bool()
                else:
                    train_mask = data.train_mask.bool()
                    test_mask = data.test_mask.bool()

                clf = LogisticRegression(
                    solver='lbfgs', random_state=self.config.RANDOM_STATE
                ).fit(z[train_mask].cpu().numpy(), data.y[train_mask].cpu().numpy())

                test_acc = accuracy_score(data.y[test_mask].cpu().numpy(), clf.predict(z[test_mask].cpu().numpy()))

                print(f"  ✅ Test Accuracy for {model_name} on {dataset_name}: {test_acc:.4f}")
                mlflow.log_metric("test_accuracy", test_acc)
                return {
                    "dataset": dataset_name,
                    "model": model_name,
                    "test_accuracy": test_acc,
                    "error": None
                }
            except Exception as e:
                print(f"  ❌ FAILED to benchmark on {dataset_name}: {e}")
                traceback.print_exc()
                mlflow.set_tag("status", "FAILED")
                mlflow.log_param("error", str(e))
                return {
                    "dataset": dataset_name,
                    "model": model_name,
                    "test_accuracy": None,
                    "error": str(e)
                }

    def run(self):
        # FIX: Suppress the noisy but harmless warning from the tokenizers library when using a multi-process DataLoader.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        DataUtils.print_header("PIPELINE: Network Embedding BENCHMARKER")
        all_results = []
        for dataset_name in self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            dataset = self._get_dataset(dataset_name)
            if not dataset:
                print(f"Skipping NE benchmark for {dataset_name} as it could not be loaded.")
                all_results.append({
                    "dataset": dataset_name, "model": "N/A", "test_accuracy": None,
                    "error": f"Dataset {dataset_name} could not be loaded."
                })
                continue

            print(
                f"\nLoaded dataset: {dataset.name}. Nodes: {dataset[0].num_nodes}, Edges: {dataset[0].num_edges}")

            for model_name in self.models_to_run:
                result = self._run_model_on_dataset(model_name, dataset)
                all_results.append(result)

        summary_df = pd.DataFrame(all_results)
        summary_path = self.output_dir / "ne_benchmark_summary.csv"
        DataUtils.save_dataframe_to_csv(summary_df, str(summary_path))
        print(f"\n==================================================")
        print(f"Network Embedding Benchmark Summary saved to: {summary_path}")
        print(summary_df.to_string())
        print(f"==============================================")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Restore default
        DataUtils.print_header("Network Embedding BENCHMARKER FINISHED")
        return summary_df