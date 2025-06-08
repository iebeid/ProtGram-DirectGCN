import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, WebKB, KarateClub  # WebKB is already imported
from torch_geometric.utils import degree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, confusion_matrix
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple

from src.config import Config
from src.models.prot_ngram_gcn import ProtNgramGCN
from src.models.gnn_zoo import get_gnn_model_from_zoo, BaseGNN


def _get_benchmark_datasets(config: Config) -> List[Tuple[torch.Tensor, str]]:
    """Loads standard benchmark datasets from PyTorch Geometric."""
    # --- MODIFICATION START ---
    # Added "Cornell" to the list of datasets to load.
    datasets_to_load = [{"name": "Cora", "class": Planetoid}, {"name": "CiteSeer", "class": Planetoid}, {"name": "PubMed", "class": Planetoid}, {"name": "Texas", "class": WebKB}, {"name": "Wisconsin", "class": WebKB},
        {"name": "Cornell", "class": WebKB},  # This line was added
        {"name": "KarateClub", "class": KarateClub}]
    # --- MODIFICATION END ---

    datasets = []
    for item in datasets_to_load:
        try:
            path = os.path.join(config.BASE_DATA_DIR, 'benchmark_datasets', item["name"])
            dataset = item["class"](root=path, name=item["name"]) if item["class"] != KarateClub else item["class"]()
            data = dataset[0]
            # Create pseudo-coordinates for MoNet using node degrees
            data.pseudo_coords = degree(data.edge_index[0], data.num_nodes).view(-1, 1).float()
            datasets.append((data, item["name"]))
            print(f"Loaded benchmark dataset: {item['name']}")
        except Exception as e:
            print(f"Could not load benchmark dataset {item['name']}. Error: {e}")
    return datasets


def _plot_confusion_matrix(y_true, y_pred, dataset_name, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred);
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues');
    plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
    plt.ylabel('Actual');
    plt.xlabel('Predicted');
    filename = f"confusion_matrix_{dataset_name}_{model_name}.png"
    plt.savefig(os.path.join(output_dir, filename));
    plt.close()


def _train_and_evaluate_fold(model, data, train_idx, test_idx, device, config, dataset_name, model_name):
    """Trains and evaluates a single fold and returns metrics."""
    model.reset_parameters() if hasattr(model, 'reset_parameters') else None
    model.to(device);
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.GCN_LR, weight_decay=config.GCN_WEIGHT_DECAY)
    best_val_loss, patience_counter, best_epoch = float('inf'), 0, 0

    for epoch in range(1, config.GCN_EPOCHS_PER_LEVEL + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(x=data.x, edge_index=data.edge_index, edge_type=getattr(data, 'edge_type', None), pseudo_coords=getattr(data, 'pseudo_coords', None))
        loss = F.nll_loss(out[train_idx], data.y[train_idx])
        loss.backward();
        optimizer.step()

        # Early stopping logic
        model.eval()
        with torch.no_grad():
            val_out, _ = model(x=data.x, edge_index=data.edge_index, edge_type=getattr(data, 'edge_type', None), pseudo_coords=getattr(data, 'pseudo_coords', None))
            val_loss = F.nll_loss(val_out[test_idx], data.y[test_idx])
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
        else:
            patience_counter += 1
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"    Early stopping at epoch {epoch}. Best epoch: {best_epoch}");
            break

    model.eval()
    with torch.no_grad():
        pred_logits, _ = model(x=data.x, edge_index=data.edge_index, edge_type=getattr(data, 'edge_type', None), pseudo_coords=getattr(data, 'pseudo_coords', None))
        pred_cpu = pred_logits[test_idx].argmax(dim=1).cpu();
        true_cpu = data.y[test_idx].cpu()
        metrics = {'accuracy': accuracy_score(true_cpu, pred_cpu), 'f1_weighted': f1_score(true_cpu, pred_cpu, average='weighted', zero_division=0),
                   'precision_weighted': precision_score(true_cpu, pred_cpu, average='weighted', zero_division=0), 'recall_weighted': recall_score(true_cpu, pred_cpu, average='weighted', zero_division=0),
                   'mcc': matthews_corrcoef(true_cpu, pred_cpu), 'convergence_epoch': best_epoch}
    return metrics


def run_gnn_benchmarking(config: Config):
    """Main entry point for the GNN benchmarking pipeline step."""
    print("\n" + "=" * 80);
    print("### PIPELINE STEP: Benchmarking GNN Architectures ###");
    print("=" * 80)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = _get_benchmark_datasets(config)
    if not datasets: print("No benchmark datasets loaded. Skipping benchmarking."); return

    models_to_benchmark = ["GCN", "GAT", "GraphSAGE", "GIN", "RGCN", "ProtNgramGCN"]
    all_results = []

    for data, dataset_name in datasets:
        print(f"\n--- Benchmarking on Dataset: {dataset_name} ---")
        num_features = data.num_features;
        num_classes = data.y.max().item() + 1

        for model_name in models_to_benchmark:
            # --- MLFLOW INTEGRATION ---
            run_name = f"{dataset_name}_{model_name}"
            with mlflow.start_run(run_name=run_name, nested=True) as run:
                print(f"  Testing model: {model_name}...")

                if config.USE_MLFLOW:
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_param("dataset_name", dataset_name)
                    mlflow.log_param("learning_rate", config.GCN_LR)
                    mlflow.log_param("epochs", config.GCN_EPOCHS_PER_LEVEL)
                    mlflow.log_param("n_folds", config.EVAL_N_FOLDS)

                fold_metrics = []
                kf = StratifiedKFold(n_splits=config.EVAL_N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
                for fold, (train_idx, test_idx) in enumerate(kf.split(np.zeros(data.num_nodes), data.y.cpu())):
                    try:
                        model = ProtNgramGCN(num_initial_features=num_features, hidden_dim1=config.GCN_HIDDEN_DIM_1, hidden_dim2=config.GCN_HIDDEN_DIM_2, num_graph_nodes=data.num_nodes,
                                             task_num_output_classes=num_classes,
                                             n_gram_len=1, one_gram_dim=0, max_pe_len=0, dropout=config.GCN_DROPOUT, use_vector_coeffs=True) if model_name == "ProtNgramGCN" else get_gnn_model_from_zoo(model_name,
                                                                                                                                                                                                         num_features,
                                                                                                                                                                                                         num_classes)
                        metrics = _train_and_evaluate_fold(model, data, train_idx, test_idx, device, config, dataset_name, model_name)
                        fold_metrics.append(metrics)
                    except Exception as e:
                        print(f"    - ERROR on fold {fold + 1} for {model_name}: {e}");
                        break

                if fold_metrics:
                    df = pd.DataFrame(fold_metrics)
                    mean_metrics = df.mean().add_suffix('_mean')
                    std_metrics = df.std().add_suffix('_std')

                    if config.USE_MLFLOW:
                        # Log mean and std metrics to MLflow
                        mlflow.log_metrics(mean_metrics.to_dict())
                        mlflow.log_metrics(std_metrics.to_dict())

                    result_row = {'dataset': dataset_name, 'model': model_name, **mean_metrics, **std_metrics}
                    all_results.append(result_row)
                    print(f"    - Avg Results: Acc={result_row['accuracy_mean']:.4f}±{result_row['accuracy_std']:.4f}, F1={result_row['f1_weighted_mean']:.4f}±{result_row['f1_weighted_std']:.4f}")

    if all_results:
        results_df = pd.DataFrame(all_results).round(4)
        print("\n--- Benchmark Results Summary ---");
        print(results_df.to_string(index=False))
        output_path = os.path.join(config.BENCHMARKING_RESULTS_DIR, "gnn_benchmark_summary.txt")
        os.makedirs(config.BENCHMARKING_RESULTS_DIR, exist_ok=True)
        with open(output_path, 'w') as f: f.write("--- GNN Benchmark Results Summary ---\n"); f.write(results_df.to_string(index=False))
        print(f"Benchmark summary saved to: {output_path}")

    print("\n### GNN Benchmarking FINISHED ###")
