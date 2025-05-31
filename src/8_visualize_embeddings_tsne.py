import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from typing import Optional, Union, List

# --- User Configuration ---
DATASET_TAG = "uniref50_visualization_v3"  # Updated tag for this version
# INPUT_H5_PATH = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/embeddings/GlobalCharGraph_Dir_UserCustomDiGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat.h5"
INPUT_H5_PATH = "C:/tmp/Models/word2vec_per_residue/word2vec_per_residue_embeddings_uniref50_w2v_per_residue_memfix_dummy.h5"
PLOTS_SAVE_DIR = "C:/tmp/Models/figures_tsne_v3"

TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
TSNE_RANDOM_STATE = 42
TSNE_INIT_PCA = True
TSNE_LEARNING_RATE = 'auto'

EMBEDDING_TYPE_IN_H5 = 'generic_multiple_sets'

MAX_PROTEINS_FOR_PER_RESIDUE_PLOT = 3
SAMPLE_N_FOR_COMBINED_TSNE = 2000


# --- End User Configuration ---

def plot_tsne_generic(embeddings_array: np.ndarray,
                      labels_for_coloring: Optional[np.ndarray],
                      title: str,
                      output_plot_path: str,
                      perplexity: float,
                      n_iter: int,
                      random_state: int,
                      init_pca: bool,
                      learning_rate: Union[str, float]):
    print(f"Generating t-SNE plot: {title}")
    num_samples = embeddings_array.shape[0]
    if num_samples == 0:
        print("  Skipping t-SNE: no data points.")
        return

    effective_perplexity = float(min(perplexity, max(1.0, num_samples - 1.0)))
    if num_samples <= effective_perplexity or num_samples <= 1:
        print(f"  Skipping t-SNE: n_samples ({num_samples}) is too small for perplexity {effective_perplexity:.1f}.")
        return
    if abs(effective_perplexity - perplexity) > 1e-5 and int(effective_perplexity) != int(perplexity):
        print(f"  Info: Adjusted t-SNE perplexity from {perplexity} to {effective_perplexity:.1f} due to n_samples.")

    tsne_init_method = 'pca' if init_pca and embeddings_array.shape[1] > 1 else 'random'
    if init_pca and embeddings_array.shape[1] <= 1:
        print("  Warning: PCA initialization for t-SNE skipped as number of features <= 1.")
        tsne_init_method = 'random'

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=effective_perplexity,
                n_iter=n_iter, init=tsne_init_method, learning_rate=learning_rate, n_jobs=-1)
    try:
        print(
            f"  Fitting t-SNE (n_samples={num_samples}, perplexity={effective_perplexity:.1f}, init={tsne_init_method})...")
        tsne_results = tsne.fit_transform(embeddings_array)
    except Exception as e:
        print(f"    Error during t-SNE transformation for '{title}': {e}. Skipping.")
        return

    df_tsne = pd.DataFrame({'tsne_1': tsne_results[:, 0], 'tsne_2': tsne_results[:, 1]})
    hue_column = None
    if labels_for_coloring is not None and len(labels_for_coloring) == num_samples:
        df_tsne['label'] = labels_for_coloring
        hue_column = 'label'
        print(f"  Using provided labels for coloring ({len(np.unique(labels_for_coloring))} unique).")

    plt.figure(figsize=(14, 12))
    unique_labels_for_palette = []
    if hue_column and hue_column in df_tsne:
        df_tsne[hue_column] = df_tsne[hue_column].astype(str)  # Ensure labels are strings for palette
        unique_labels_for_palette = sorted(df_tsne[hue_column].unique())

    palette = None
    if unique_labels_for_palette:
        if 0 < len(unique_labels_for_palette) <= 10:
            palette = sns.color_palette("Paired", n_colors=len(unique_labels_for_palette))
        elif 10 < len(unique_labels_for_palette) <= 20:
            palette = sns.color_palette("tab20", n_colors=len(unique_labels_for_palette))
        else:
            palette = sns.color_palette("husl", n_colors=len(unique_labels_for_palette))

    scatter_plot = sns.scatterplot(x="tsne_1", y="tsne_2", hue=hue_column, data=df_tsne,
                                   legend="full" if hue_column and len(unique_labels_for_palette) <= 25 else False,
                                   palette=palette, s=50, alpha=0.6)
    plt.title(title, fontsize=16);
    plt.xlabel('t-SNE Component 1', fontsize=12);
    plt.ylabel('t-SNE Component 2', fontsize=12)

    if hue_column and unique_labels_for_palette and len(unique_labels_for_palette) <= 25:
        handles, current_labels_from_plot = scatter_plot.get_legend_handles_labels()
        if len(handles) > 0: plt.legend(title=str(hue_column).capitalize(), bbox_to_anchor=(1.02, 1), loc='upper left',
                                        borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    else:
        if hue_column and len(unique_labels_for_palette) > 25: print("  Legend omitted due to >25 unique labels.")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_plot_path = os.path.normpath(output_plot_path)
    plot_parent_dir = os.path.dirname(output_plot_path)
    try:
        if not os.path.exists(plot_parent_dir): os.makedirs(plot_parent_dir, exist_ok=True)
        plt.savefig(output_plot_path, dpi=150);
        print(f"  t-SNE plot saved to {output_plot_path}")
    except Exception as e:
        print(f"    Error saving t-SNE plot to {output_plot_path}: {e}")
    plt.close()


def main():
    print("--- Script: Generic t-SNE Visualization for HDF5 Embeddings ---")
    os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)
    if not os.path.isdir(PLOTS_SAVE_DIR): print(f"Error: Output directory {PLOTS_SAVE_DIR} error. Exiting."); return

    norm_input_h5_path = os.path.normpath(INPUT_H5_PATH)
    if not os.path.exists(norm_input_h5_path): print(
        f"Error: Input HDF5 file not found: {norm_input_h5_path}. Exiting."); return

    print(f"Loading embeddings from: {norm_input_h5_path}")
    print(f"Interpreting embeddings as: {EMBEDDING_TYPE_IN_H5}")
    base_filename = os.path.splitext(os.path.basename(norm_input_h5_path))[0]

    with h5py.File(norm_input_h5_path, 'r') as hf:
        all_keys = list(hf.keys())
        if not all_keys: print("No datasets found in HDF5 file."); return
        print(f"Found {len(all_keys)} top-level items/datasets in HDF5 file.")

        if EMBEDDING_TYPE_IN_H5 == 'per_residue':
            protein_ids_to_process = all_keys[:min(len(all_keys), MAX_PROTEINS_FOR_PER_RESIDUE_PLOT)]
            if not protein_ids_to_process: print("No protein IDs for per-residue plotting."); return
            for prot_id in tqdm(protein_ids_to_process, desc="Plotting per-residue t-SNE"):
                if prot_id not in hf: print(f"Warning: Key {prot_id} not in H5. Skipping."); continue
                per_residue_embeddings = hf[prot_id][:]
                if per_residue_embeddings.ndim != 2 or per_residue_embeddings.shape[0] == 0:
                    print(f"Skipping {prot_id}: Invalid embedding shape {per_residue_embeddings.shape}");
                    continue
                plot_title = f"t-SNE of Per-Residue Embeddings for {prot_id}\n(Source: {base_filename})"
                safe_prot_id = prot_id.replace('|', '_').replace('/', '_').replace(':', '_')
                plot_path = os.path.join(PLOTS_SAVE_DIR,
                                         f"tsne_residues_{base_filename}_{safe_prot_id}_{DATASET_TAG}.png")
                plot_tsne_generic(embeddings_array=per_residue_embeddings, labels_for_coloring=None, title=plot_title,
                                  output_plot_path=plot_path, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_N_ITER,
                                  random_state=TSNE_RANDOM_STATE, init_pca=TSNE_INIT_PCA,
                                  learning_rate=TSNE_LEARNING_RATE)

        elif EMBEDDING_TYPE_IN_H5 == 'per_protein' or EMBEDDING_TYPE_IN_H5 == 'generic_multiple_sets':
            embeddings_list = [];
            labels_list = []
            desc_text = "Loading per-protein embeddings" if EMBEDDING_TYPE_IN_H5 == 'per_protein' else "Loading generic embedding sets"

            print(f"\nInspecting datasets for '{EMBEDDING_TYPE_IN_H5}' mode...")  # Added print
            for key_id in tqdm(all_keys, desc=desc_text):
                if key_id not in hf:
                    print(f"  Warning: Key {key_id} from initial scan not found during iteration. Skipping.")
                    continue
                dataset_obj = hf[key_id]
                if isinstance(dataset_obj, h5py.Dataset):
                    data_array = dataset_obj[:]
                    if EMBEDDING_TYPE_IN_H5 == 'per_protein':
                        if data_array.ndim == 1 and data_array.shape[0] > 0:
                            embeddings_list.append(data_array)
                        elif data_array.ndim == 2 and data_array.shape[0] == 1 and data_array.shape[1] > 0:
                            embeddings_list.append(data_array.flatten())
                        else:
                            print(
                                f"  Skipping dataset '{key_id}' for per_protein: Not a valid 1D or (1,D) embedding (shape: {data_array.shape}).")  # Diagnostic
                    elif EMBEDDING_TYPE_IN_H5 == 'generic_multiple_sets':
                        is_valid_1d = data_array.ndim == 1 and data_array.shape[0] > 0
                        is_valid_2d = data_array.ndim == 2 and data_array.shape[0] > 0 and data_array.shape[1] > 0
                        if is_valid_1d:
                            embeddings_list.append(data_array.reshape(1, -1))
                            labels_list.append(key_id)
                        elif is_valid_2d:
                            embeddings_list.append(data_array)
                            labels_list.extend([key_id] * data_array.shape[0])
                        else:
                            print(
                                f"  Skipping dataset '{key_id}' for generic_multiple_sets: Not a valid 1D or 2D embedding array with non-zero dimensions (shape: {data_array.shape}).")
                else:
                    print(f"  Skipping key '{key_id}': It is an HDF5 Group, not a Dataset.")

            print(f"Finished inspecting datasets. Found {len(embeddings_list)} valid embedding arrays/sets to combine.")
            if not embeddings_list: print("No valid embeddings loaded for combined plot."); return

            try:
                embeddings_array = np.vstack(embeddings_list)
            except ValueError as ve:
                print(f"Error combining embeddings (likely inconsistent feature dimensions): {ve}");
                return

            labels_for_coloring = np.array(
                labels_list) if EMBEDDING_TYPE_IN_H5 == 'generic_multiple_sets' and labels_list else None

            if embeddings_array.shape[0] > SAMPLE_N_FOR_COMBINED_TSNE:
                print(f"Sampling {SAMPLE_N_FOR_COMBINED_TSNE} embeddings from {embeddings_array.shape[0]} for t-SNE.")
                indices = np.random.choice(embeddings_array.shape[0], SAMPLE_N_FOR_COMBINED_TSNE, replace=False)
                embeddings_array = embeddings_array[indices]
                if labels_for_coloring is not None: labels_for_coloring = labels_for_coloring[indices]

            plot_title_prefix = "Per-Protein Embeddings" if EMBEDDING_TYPE_IN_H5 == 'per_protein' else "Generic Embedding Sets"
            plot_title = f"t-SNE of {plot_title_prefix}\n(Source: {base_filename})"
            plot_filename_suffix = "proteins" if EMBEDDING_TYPE_IN_H5 == 'per_protein' else "generic_sets"
            plot_path = os.path.join(PLOTS_SAVE_DIR, f"tsne_{plot_filename_suffix}_{base_filename}_{DATASET_TAG}.png")
            plot_tsne_generic(embeddings_array=embeddings_array, labels_for_coloring=labels_for_coloring,
                              title=plot_title,
                              output_plot_path=plot_path, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_N_ITER,
                              random_state=TSNE_RANDOM_STATE, init_pca=TSNE_INIT_PCA, learning_rate=TSNE_LEARNING_RATE)
        else:
            print(f"Error: Unknown EMBEDDING_TYPE_IN_H5: '{EMBEDDING_TYPE_IN_H5}'.")
    print("Script finished.")


if __name__ == "__main__":
    main()