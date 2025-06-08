# ==============================================================================
# MODULE: benchmarking/gnn_evaluator.py
# PURPOSE: Contains the workflow for benchmarking GNN models, including our
#          custom ProtNgramGCN, on standard academic graph datasets.
# VERSION: 2.0 (With expanded metrics and models)
# ==============================================================================

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

# Import from our new project structure
from src.config import Config
from src.models.prot_ngram_gcn import ProtNgramGCN
from src.models.gnn_zoo import get_gnn_model_from_zoo, BaseGNN


def _get_benchmark_datasets(config: Config) -> List[Tuple[Data, str]]:
    """Loads standard benchmark datasets from PyTorch Geometric."""
    dataset_names = ["Cora", "CiteSeer", "PubMed"]
    datasets = []
    for name in dataset_names:
        try:
            dataset_path = os.path.join(config.BASE_DATA_DIR, 'benchmark_datasets', name)
            dataset = Planetoid(root=dataset_path, name=name)
            datasets.append((dataset[0], name))
            print(f"Loaded benchmark dataset: {name}")
        except Exception as e:
            print(f"Could not load benchmark dataset {name}. Error: {e}")
    return datasets


def _plot_confusion_matrix(y_true, y_pred, class_names, dataset_name, model_name, output_dir):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    filename = f"confusion_matrix_{dataset_name}_{model_name}.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def _train_and_evaluate_model(
        model: BaseGNN,
        data: Data,
        device: torch.device,
        config: Config,
        dataset_name: str
) -> Dict[str, Any]:
    """
    Trains and evaluates a single GNN model on a given dataset's splits.
    Returns a dictionary of performance metrics.
    """
    model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.GCN_LR, weight_decay=config.GCN_WEIGHT_DECAY)

    model.train()
    for epoch in range(config.GCN_EPOCHS_PER_LEVEL):
        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_logits, _ = model(data.x, data.edge_index)
        pred_cpu = pred_logits[data.test_mask].argmax(dim=1).cpu()
        true_cpu = data.y[data.test_mask].cpu()

        # Generate and save confusion matrix plot
        class_names = [str(i) for i in range(data.y.max().item() + 1)]
        plots_dir = os.path.join(config.BENCHMARKING_RESULTS_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        _plot_confusion_matrix(true_cpu, pred_cpu, class_names, dataset_name, model.name, plots_dir)

        # Calculate all metrics
        metrics = {
            'accuracy': accuracy_score(true_cpu, pred_cpu),
            'f1_weighted': f1_score(true_cpu, pred_cpu, average='weighted', zero_division=0),
            'precision_weighted': precision_score(true_cpu, pred_cpu, average='weighted', zero_division=0),
            'recall_weighted': recall_score(true_cpu, pred_cpu, average='weighted', zero_division=0),
            'mcc': matthews_corrcoef(true_cpu, pred_cpu)
        }
        return metrics


def run_gnn_benchmarking(config: Config):
    """
    The main entry point for the GNN benchmarking pipeline step.
    """
    print("\n" + "=" * 80)
    print("### PIPELINE STEP: Benchmarking GNN Architectures ###")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = _get_benchmark_datasets(config)
    if not datasets:
        print("No benchmark datasets loaded. Skipping benchmarking.")
        return

    # Define the models to benchmark, now including RGCN
    models_to_benchmark = ["GCN", "GAT", "GraphSAGE", "GIN", "RGCN", "ProtNgramGCN"]
    all_results = []

    for data, dataset_name in datasets:
        print(f"\n--- Benchmarking on Dataset: {dataset_name} ---")
        num_features = data.num_features
        num_classes = data.y.max().item() + 1
        num_relations = data.num_edge_types if hasattr(data, 'num_edge_types') and data.num_edge_types else 1

        for model_name in models_to_benchmark:
            print(f"  Testing model: {model_name}...")

            try:
                if model_name == "ProtNgramGCN":
                    model = ProtNgramGCN(
                        num_initial_features=num_features, hidden_dim1=config.GCN_HIDDEN_DIM_1,
                        hidden_dim2=config.GCN_HIDDEN_DIM_2, num_graph_nodes=data.num_nodes,
                        task_num_output_classes=num_classes, n_gram_len=3, one_gram_dim=0, max_pe_len=0,
                        dropout=config.GCN_DROPOUT, use_vector_coeffs=True)
                else:
                    model = get_gnn_model_from_zoo(model_name, num_features, num_classes, num_relations=num_relations)

                model.name = model_name  # Assign name for plotting
                metrics = _train_and_evaluate_model(model, data, device, config, dataset_name)

                result_row = {'dataset': dataset_name, 'model': model_name, **metrics}
                all_results.append(result_row)
                print(f"    - Results: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_weighted']:.4f}, MCC={metrics['mcc']:.4f}")
            except Exception as e:
                print(f"    - ERROR running {model_name} on {dataset_name}: {e}")

    # --- Reporting Results ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        # Reorder columns for clarity
        cols_order = ['dataset', 'model', 'accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'mcc']
        results_df = results_df[[c for c in cols_order if c in results_df.columns]]

        print("\n--- Benchmark Results Summary ---")
        print(results_df.to_string(index=False))

        output_path = os.path.join(config.BENCHMARKING_RESULTS_DIR, "gnn_benchmark_summary.txt")
        os.makedirs(config.BENCHMARKING_RESULTS_DIR, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("--- GNN Benchmark Results Summary ---\n")
            f.write(results_df.to_string(index=False))
        print(f"Benchmark summary saved to: {output_path}")

    print("\n### GNN Benchmarking FINISHED ###")