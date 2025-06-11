# ==============================================================================
# MODULE: gnn_benchmarker.py
# PURPOSE: To benchmark various GNN models on standard datasets.
# VERSION: 2.3 (Added GINConv)
# ==============================================================================

import os
import pandas as pd
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

# Assuming these are correctly located in your project structure
from src.models.protgram_directgcn import ProtNgramGCN
from src.models.gnn_zoo import *
from src.config import Config
from src.utils.data_utils import DataUtils


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
            for data_batch in val_loader:  # Renamed to avoid conflict with outer 'data'
                data_batch = data_batch.to(device)
                preds = model(data_batch)
                all_preds.append(preds)
                all_labels.append(data_batch.y)
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
        for data_batch in test_loader:  # Renamed to avoid conflict with outer 'data'
            data_batch = data_batch.to(device)
            preds = model(data_batch)
            all_preds.append(preds)
            all_labels.append(data_batch.y)
        test_f1 = f1_score(torch.cat(all_labels).cpu().numpy(), (torch.cat(all_preds) > 0).cpu().numpy(), average='micro')

    return best_f1, test_f1, pd.DataFrame(history)


# ==============================================================================
# Benchmarking Pipeline
# ==============================================================================

def run_benchmarker(config: Config):
    """Runs the GNN benchmarking pipeline."""
    DataUtils.print_header("PIPELINE: GNN BENCHMARKER")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    path = os.path.join(config.BASE_INPUT_DIR, 'PPI')
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')

    train_loader = DataLoader(train_dataset, batch_size=config.GCN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)  # PPI val/test batch_size is usually 2
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)  # PPI val/test batch_size is usually 2

    num_features = train_dataset.num_features
    num_classes = train_dataset.num_classes

    # Updated model_names list
    model_names = ["GCN", "GAT", "GraphSAGE", "WLGCN", "CHEBNET", "SIGNEDNET", "RGCN_SR", "GIN", "ProtNgramGCN"]
    results = []

    for model_name in model_names:
        print(f"\n--- Benchmarking Model: {model_name} ---")

        # =================================================================================
        # ✨ FIXED: ProtNgramGCN instantiation now correctly uses parameters from the
        # Config object, mirroring the logic in protgram_directgcn_embedder.py.
        # =================================================================================
        if model_name == "ProtNgramGCN":
            # Ensure ProtNgramGCN specific parameters are correctly handled if they differ
            # For example, num_graph_nodes might not be applicable for PPI if it varies.
            model = ProtNgramGCN(num_initial_features=num_features, hidden_dim1=config.GCN_HIDDEN_DIM_1, hidden_dim2=config.GCN_HIDDEN_DIM_2, num_graph_nodes=None,  # PPI graphs have varying node counts
                                 task_num_output_classes=num_classes, n_gram_len=config.GCN_NGRAM_MAX_N, one_gram_dim=config.GCN_ONE_GRAM_EMBED_DIM, max_pe_len=config.GCN_MAX_PE_LEN, dropout=config.GCN_DROPOUT,
                                 use_vector_coeffs=config.GCN_USE_VECTOR_COEFFS)
        else:
            model = get_gnn_model_from_zoo(model_name, num_features, num_classes)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.GCN_LEARNING_RATE)

        val_f1, test_f1, history_df = train_and_evaluate(model, train_loader, val_loader, test_loader, optimizer, device, num_epochs=config.GCN_NUM_EPOCHS if hasattr(config, 'GCN_NUM_EPOCHS') else 200)

        print(f"Final Results for {model_name}: Best Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}")
        results.append({'model': model_name, 'best_val_f1': val_f1, 'test_f1': test_f1})

        # Ensure reports directory exists
        reports_dir = config.REPORTS_DIR
        os.makedirs(reports_dir, exist_ok=True)

        history_path = os.path.join(reports_dir, f'benchmark_{model_name}_history.csv')
        history_df.to_csv(history_path, index=False)
        print(f"Saved {model_name} training history to {history_path}")

    # Save overall results
    results_df = pd.DataFrame(results)
    DataUtils.save_dataframe_to_csv(results_df, config.REPORTS_DIR, "gnn_benchmark_summary.csv")
    DataUtils.print_header("GNN Benchmarking FINISHED")
