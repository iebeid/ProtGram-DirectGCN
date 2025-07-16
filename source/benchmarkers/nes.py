# ==============================================================================
# MODULE: benchmarkers/nes.py
# PURPOSE: Benchmarks traditional Network Embedding models like Node2Vec.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import time
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
from torch_geometric.datasets import (KarateClub, Planetoid)

from configuration.config import Config
from source.utils.data import DataUtils


class NetworkEmbeddingBenchmarker:
    """
    A benchmarker for traditional network embedding models on node classification tasks.
    It first learns embeddings in an unsupervised manner, then trains a logistic
    regression classifier on top of these embeddings for evaluation.
    """
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = []

    def _get_dataset(self, name: str):
        """Loads a dataset from PyG."""
        path = self.config.DATA_STANDARD_DATASETS_DIR / name
        if name in ["Cora", "CiteSeer", "PubMed"]:
            return Planetoid(root=str(path), name=name)
        elif name == "KarateClub":
            return KarateClub()
        elif name in ["Cornell", "Texas", "Wisconsin"]:
            # These are part of the "Heterophilous" collection in PyG
            from torch_geometric.datasets import HeterophilousGraphDataset
            return HeterophilousGraphDataset(root=str(path), name=name)
        raise ValueError(f"Unknown dataset: {name}")

    def run_single_benchmark(self, model_name: str, data):
        """Runs a benchmark for a single model on a given dataset."""
        print(f"--- Benchmarking Model: {model_name} on Dataset: {data.__class__.__name__} ---")

        # --- 1. Unsupervised Embedding Generation ---
        if model_name == "Node2Vec":
            model = Node2Vec(data.edge_index, embedding_dim=self.config.BENCHMARK_NE_EMBEDDING_DIM,
                             walk_length=self.config.BENCHMARK_NE_WALK_LENGTH,
                             context_size=self.config.BENCHMARK_NE_CONTEXT_SIZE,
                             walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True).to(self.device)
        elif model_name == "DeepWalk":
            model = Node2Vec(data.edge_index, embedding_dim=self.config.BENCHMARK_NE_EMBEDDING_DIM,
                             walk_length=self.config.BENCHMARK_NE_WALK_LENGTH,
                             context_size=self.config.BENCHMARK_NE_CONTEXT_SIZE,
                             walks_per_node=10, num_negative_samples=1, p=1, q=1, sparse=True).to(self.device) # DeepWalk is Node2Vec with p=1, q=1
        else:
            raise ValueError(f"Unsupported Network Embedding model: {model_name}")

        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        model.train()
        for _ in range(self.config.EVAL_EPOCHS): # Use existing epochs config
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()

        # --- 2. Downstream Classification Task ---
        model.eval()
        with torch.no_grad():
            z = model()

        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

        # Handle datasets that may not have standard splits
        if train_mask is None or val_mask is None or test_mask is None:
            print("  Dataset has no standard splits. Creating random splits...")
            # This is a simplified split logic, you might want to use the one from GNNBenchmarker
            num_nodes = data.num_nodes
            indices = torch.randperm(num_nodes)
            train_size = int(num_nodes * 0.6)
            val_size = int(num_nodes * 0.2)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size+val_size]] = True
            test_mask[indices[train_size+val_size:]] = True

        classifier = LogisticRegression(random_state=self.config.RANDOM_STATE, max_iter=1000)
        classifier.fit(z[train_mask].cpu().numpy(), data.y[train_mask].cpu().numpy())

        val_acc = classifier.score(z[val_mask].cpu().numpy(), data.y[val_mask].cpu().numpy())
        test_acc = classifier.score(z[test_mask].cpu().numpy(), data.y[test_mask].cpu().numpy())

        print(f"  Val Accuracy: {val_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        self.results.append({
            "dataset": data.__class__.__name__,
            "model": model_name,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc
        })

    def run(self):
        """Main execution method for the benchmarker."""
        DataUtils.print_header("PIPELINE: Network Embedding BENCHMARKER")

        for dataset_name in self.config.BENCHMARK_NODE_CLASSIFICATION_DATASETS:
            try:
                dataset = self._get_dataset(dataset_name)
                data = dataset[0]
                print(f"\nLoaded dataset: {dataset_name}. Nodes: {data.num_nodes}, Edges: {data.num_edges}")

                for model_name in self.config.BENCHMARK_NE_MODELS_TO_RUN:
                    self.run_single_benchmark(model_name, data)

            except Exception as e:
                print(f"  ❌ FAILED to benchmark on {dataset_name}: {e}")
                continue

        summary_df = pd.DataFrame(self.results)
        output_path = self.config.RESULTS_BENCHMARKING_DIR / "ne_benchmark_summary.csv"
        summary_df.to_csv(output_path, index=False)
        print("\n" + "=" * 50)
        print(f"Network Embedding Benchmark Summary saved to: {output_path}")
        print(summary_df)
        DataUtils.print_header("Network Embedding BENCHMARKER FINISHED")