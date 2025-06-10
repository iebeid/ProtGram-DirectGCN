# ==============================================================================
# MODULE: gnn_benchmarker.py
# PURPOSE: To benchmark various GNN models on standard datasets.
# VERSION: 2.2 (Corrected, Synthesized, and Self-Contained)
# ==============================================================================

import os
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# Assuming these are correctly located in your project structure
from src.models.prot_ngram_gcn import ProtNgramGCN
from src.config import Config


# ==============================================================================
# Helper Functions (Implemented directly to avoid import errors)
# ==============================================================================

def print_header(title):
    """Prints a formatted header to the console."""
    border = "=" * (len(title) + 4)
    print(f"\n{border}\n### {title} ###\n{border}\n")


def save_benchmark_results(results_df, output_dir):
    """Saves the benchmark results dataframe to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'gnn_benchmark_summary.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nBenchmark summary saved to: {results_path}")


def get_gnn_model_from_zoo(model_name, num_features, num_classes):
    """
    A simple factory to get standard GNN models from PyG.
    This replaces the need for a separate gnn_zoo.py file for this script.
    """
    if model_name.upper() == 'GCN':
        return GCN(in_channels=num_features, hidden_channels=256, out_channels=num_classes)
    elif model_name.upper() == 'GAT':
        return GAT(in_channels=num_features, hidden_channels=256, out_channels=num_classes, heads=4)
    elif model_name.upper() == 'GRAPHSAGE':
        return GraphSAGE(in_channels=num_features, hidden_channels=256, out_channels=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported in the local GNN Zoo.")


# --- Basic PyG Model Implementations for the Zoo ---
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.sage2(x, edge_index)
        return x


# ==============================================================================
# Main Training & Evaluation Logic
# ==============================================================================

def train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, device, num_epochs=200):
    """Main training and evaluation loop."""
    best_f1 = 0
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.binary_cross_entropy_with_logits(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for data in val_loader:
                data = data.to(device)
                preds = model(data)
                all_preds.append(preds)
                all_labels.append(data.y)
            val_f1 = f1_score(torch.cat(all_labels).cpu().numpy(), (torch.cat(all_preds) > 0).cpu().numpy(), average='micro')

        if val_f1 > best_f1:
            best_f1 = val_f1

        print(f'Epoch {epoch:03d}, Loss: {total_loss / len(train_loader):.4f}, Val F1: {val_f1:.4f}')
        history.append({'epoch': epoch, 'loss': total_loss / len(train_loader), 'val_f1': val_f1})

    # Final Test Evaluation
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for data in test_loader:
            data = data.to(device)
            preds = model(data)
            all_preds.append(preds)
            all_labels.append(data.y)
        test_f1 = f1_score(torch.cat(all_labels).cpu().numpy(), (torch.cat(all_preds) > 0).cpu().numpy(), average='micro')

    return best_f1, test_f1, pd.DataFrame(history)


# ==============================================================================
# Benchmarking Pipeline
# ==============================================================================

def run_benchmarker(config: Config):
    """Runs the GNN benchmarking pipeline."""
    print_header("PIPELINE: GNN BENCHMARKER")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    path = os.path.join(config.BASE_INPUT_DIR, 'PPI')
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')

    train_loader = DataLoader(train_dataset, batch_size=config.GCN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    num_features = train_dataset.num_features
    num_classes = train_dataset.num_classes

    model_names = ["GCN", "GAT", "GraphSAGE", "ProtNgramGCN"]
    results = []

    for model_name in model_names:
        print(f"\n--- Benchmarking Model: {model_name} ---")

        # =================================================================================
        # ✨ FIXED: ProtNgramGCN instantiation now correctly uses parameters from the
        # Config object, mirroring the logic in prot_ngram_gcn_trainer.py.
        # =================================================================================
        if model_name == "ProtNgramGCN":
            model = ProtNgramGCN(num_initial_features=num_features, hidden_dim1=config.GCN_HIDDEN_DIM_1, hidden_dim2=config.GCN_HIDDEN_DIM_2, num_graph_nodes=None,
                # Set to None for PPI dataset where node count varies per graph
                task_num_output_classes=num_classes, n_gram_len=config.GCN_NGRAM_MAX_N, one_gram_dim=config.GCN_ONE_GRAM_EMBED_DIM, max_pe_len=config.GCN_MAX_PE_LEN, dropout=config.GCN_DROPOUT,
                use_vector_coeffs=config.GCN_USE_VECTOR_COEFFS)
        else:
            model = get_gnn_model_from_zoo(model_name, num_features, num_classes)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.GCN_LEARNING_RATE)

        val_f1, test_f1, history_df = train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, device)

        print(f"Final Results for {model_name}: Best Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}")
        results.append({'model': model_name, 'best_val_f1': val_f1, 'test_f1': test_f1})

        history_path = os.path.join(config.REPORTS_DIR, f'benchmark_{model_name}_history.csv')
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        history_df.to_csv(history_path, index=False)
        print(f"Saved {model_name} training history to {history_path}")

    save_benchmark_results(pd.DataFrame(results), config.REPORTS_DIR)
    print_header("GNN Benchmarking FINISHED")
