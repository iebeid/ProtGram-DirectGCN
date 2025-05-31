import pandas as pd
import numpy as np
import h5py
import os
import shutil  # For creating/removing directories
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Configuration & Dummy Data Generation ---
DUMMY_RESIDUE_VOCAB = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']  # 'X' for unknown


def create_dummy_residue_h5(file_path, embed_dim, vocab, file_idx=0):
    """Creates a dummy H5 file with residue embeddings, allowing varied dimensions."""
    with h5py.File(file_path, 'w') as hf:
        for residue_idx, residue in enumerate(vocab):
            # Vary embedding values slightly per file and residue for distinctness
            val = (ord(residue) - ord('A') + 1 + file_idx * 10 + residue_idx * 0.1) / 100.0
            embedding = np.full((embed_dim,), val, dtype=np.float32)
            if residue == 'X':
                embedding = np.zeros((embed_dim,), dtype=np.float32)
            hf.create_dataset(residue, data=embedding)
    # print(f"Dummy residue H5 created: '{os.path.basename(file_path)}' (dim={embed_dim})")


def create_dummy_protein_csv(file_path, num_proteins=5, has_header=True,
                             id_col_name='uniprot_id', seq_col_name='sequence'):
    """Creates a dummy CSV file with UniProt IDs and protein sequences."""
    data = []
    for i in range(1, num_proteins + 1):
        uniprot_id = f"P{i:05d}"
        seq_len = np.random.randint(10, 30)
        sequence = "".join(np.random.choice(DUMMY_RESIDUE_VOCAB[:-1], size=seq_len))
        if np.random.rand() < 0.1:
            pos = np.random.randint(0, len(sequence))
            sequence = sequence[:pos] + 'X' + sequence[pos + 1:]
        data.append([uniprot_id, sequence])
    df = pd.DataFrame(data)
    if has_header:
        df.columns = [id_col_name, seq_col_name]
        df.to_csv(file_path, index=False, header=True)
    else:
        df.to_csv(file_path, index=False, header=False)
    # print(f"Dummy protein CSV ({'with' if has_header else 'NO'} header) created: '{os.path.basename(file_path)}'")


# --- Residue Embedding Loading (Mostly Unchanged) ---
def load_residue_embeddings_from_h5(h5_file_path):
    if not os.path.exists(h5_file_path):
        print(f"Error: Residue embedding file not found: {h5_file_path}")
        return None, -1
    residue_embeddings = {}
    embedding_dim = -1
    try:
        with h5py.File(h5_file_path, 'r') as hf:
            if not hf.keys():
                print(f"Error: No residues in H5 file: {h5_file_path}")
                return None, -1
            for i, residue in enumerate(list(hf.keys())):
                vector = hf[residue][:]
                if i == 0:
                    embedding_dim = len(vector)
                elif len(vector) != embedding_dim:
                    print(f"Error: Inconsistent embedding dimension for '{residue}' in {os.path.basename(h5_file_path)}. "
                          f"Expected {embedding_dim}, got {len(vector)}.")
                    return None, -1  # Dimension consistency within a single file is vital
                residue_embeddings[residue] = vector
        # print(f"Loaded {len(residue_embeddings)} res. embeddings from '{os.path.basename(h5_file_path)}', dim: {embedding_dim}.")
        return residue_embeddings, embedding_dim
    except Exception as e:
        print(f"Error loading residue embeddings from {h5_file_path}: {e}")
        return None, -1


# --- Single Protein Processing (Unchanged) ---
def process_single_protein(protein_data, residue_embeddings_map, unknown_residue_key='X'):
    uniprot_id, sequence = protein_data
    if not sequence or not isinstance(sequence, str): return uniprot_id, None
    protein_residue_vectors = []
    embedding_dim = -1
    if residue_embeddings_map:
        first_known_vec = next(iter(residue_embeddings_map.values()), None)
        if first_known_vec is not None: embedding_dim = len(first_known_vec)

    for residue in sequence:
        vector = residue_embeddings_map.get(residue)
        if vector is None and unknown_residue_key: vector = residue_embeddings_map.get(unknown_residue_key)
        if vector is not None:
            if embedding_dim == -1: embedding_dim = len(vector)
            if len(vector) == embedding_dim: protein_residue_vectors.append(vector)
    if not protein_residue_vectors:
        if embedding_dim > 0: return uniprot_id, np.zeros(embedding_dim, dtype=np.float32)
        return uniprot_id, None
    mean_embedding = np.mean(protein_residue_vectors, axis=0, dtype=np.float32)
    return uniprot_id, mean_embedding


# --- PCA Application Function ---
def apply_pca_to_embeddings(embedding_vectors_list, target_dimension, original_dimension):
    """
    Applies PCA to a list of embedding vectors to reduce/transform them to target_dimension.
    Returns: Transformed embeddings as a NumPy array, or None if PCA fails.
    """
    if not embedding_vectors_list:
        print("PCA: No embedding vectors to transform.")
        return None

    embedding_matrix = np.array(embedding_vectors_list)

    if embedding_matrix.shape[0] == 0:  # No samples
        print("PCA: Embedding matrix is empty after filtering Nones.")
        return None

    # Number of samples must be >= target_dimension for meaningful PCA reduction to that target
    # Also, target_dimension must be <= original_dimension
    n_samples = embedding_matrix.shape[0]

    actual_target_dim = min(target_dimension, original_dimension, n_samples)
    if actual_target_dim < target_dimension:
        print(f"PCA: Adjusted target dimension from {target_dimension} to {actual_target_dim} due to data constraints "
              f"(n_samples={n_samples}, original_dim={original_dimension}).")
    if actual_target_dim <= 0:
        print(f"PCA: Cannot perform PCA with target dimension {actual_target_dim}. Skipping.")
        return None

    try:
        # It's good practice to scale data before PCA
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embedding_matrix)

        pca = PCA(n_components=actual_target_dim, svd_solver='full')
        transformed_embeddings = pca.fit_transform(scaled_embeddings)
        print(f"PCA: Applied. Original shape: {embedding_matrix.shape}, "
              f"Transformed shape: {transformed_embeddings.shape}. "
              f"Explained variance ratio by {actual_target_dim} components: {np.sum(pca.explained_variance_ratio_):.4f}")

        # If actual_target_dim was less than target_dimension, and we STRICTLY need target_dimension,
        # we might need to pad. However, for this implementation, we'll return the PCA result as is.
        # If strict output dimension is required even with padding:
        if transformed_embeddings.shape[1] < target_dimension:
            print(f"PCA: Output dimension {transformed_embeddings.shape[1]} is less than target {target_dimension}. "
                  "Padding with zeros to meet target dimension might be an option if strictly required, "
                  "but current implementation returns PCA output as is.")
            # Example for padding (use with caution, as it might not be meaningful):
            # padding = np.zeros((transformed_embeddings.shape[0], target_dimension - transformed_embeddings.shape[1]))
            # transformed_embeddings = np.hstack((transformed_embeddings, padding))
            # print(f"PCA: Padded to shape: {transformed_embeddings.shape}")

        return transformed_embeddings
    except ValueError as ve:
        print(f"PCA Error: Could not apply PCA (n_components={actual_target_dim}, n_samples={n_samples}, "
              f"n_features={embedding_matrix.shape[1]}). Error: {ve}. Skipping PCA for this set.")
        return None  # Or return original embeddings if preferred and dimensions match target (unlikely)


# --- Main Orchestration Function for Directory Processing ---
def process_residue_embeddings_directory(protein_csv_path, input_residue_h5_dir, output_protein_h5_dir,
                                         target_pca_dimension,
                                         uniprot_id_col='uniprot_id', sequence_col='sequence',
                                         protein_csv_header='infer', num_workers=None):
    print("--- Starting Batch Protein Embedding Generation with PCA ---")

    # 1. Prepare output directory
    if not os.path.exists(output_protein_h5_dir):
        os.makedirs(output_protein_h5_dir)
        print(f"Created output directory: '{output_protein_h5_dir}'")

    # 2. Load protein sequences (once)
    if not os.path.exists(protein_csv_path):
        print(f"Error: Protein CSV file not found: {protein_csv_path}")
        return

    protein_tasks = []
    try:  # Copied and adapted from previous generate_protein_mean_embeddings
        protein_df = pd.read_csv(protein_csv_path, header=protein_csv_header)
        protein_tasks_data_selected = None
        if protein_csv_header is None:
            try:
                uid_col_idx, seq_col_idx = int(uniprot_id_col), int(sequence_col)
                if not (0 <= uid_col_idx < len(protein_df.columns) and 0 <= seq_col_idx < len(protein_df.columns) and uid_col_idx != seq_col_idx):
                    raise ValueError("Column indices invalid or not distinct.")
                protein_tasks_data_selected = protein_df.iloc[:, [uid_col_idx, seq_col_idx]]
            except (ValueError, TypeError):
                print("Error: If 'protein_csv_header' is None, 'uniprot_id_col' and 'sequence_col' must be valid, distinct integer indices.");
                return
        else:
            if not (isinstance(uniprot_id_col, str) and isinstance(sequence_col, str)): print("Error: With header, column identifiers must be strings."); return
            if uniprot_id_col not in protein_df.columns or sequence_col not in protein_df.columns:
                print(f"Error: Columns '{uniprot_id_col}' or '{sequence_col}' not in {protein_csv_path}. Available: {protein_df.columns.tolist()}");
                return
            protein_tasks_data_selected = protein_df[[uniprot_id_col, sequence_col]]
        if protein_tasks_data_selected is not None and not protein_tasks_data_selected.empty:
            protein_tasks_data_selected.iloc[:, 1] = protein_tasks_data_selected.iloc[:, 1].astype(str).fillna('')
            protein_tasks = list(protein_tasks_data_selected.itertuples(index=False, name=None))
        if not protein_tasks: print(f"No protein data loaded from '{protein_csv_path}'."); return
        print(f"Loaded {len(protein_tasks)} protein sequences from '{protein_csv_path}'.")
    except Exception as e:
        print(f"Error reading protein CSV '{protein_csv_path}': {e}"); return

    # 3. List input residue H5 files
    if not os.path.isdir(input_residue_h5_dir):
        print(f"Error: Input residue H5 directory not found: {input_residue_h5_dir}");
        return

    residue_h5_files = [f for f in os.listdir(input_residue_h5_dir) if f.endswith('.h5') or f.endswith('.hdf5')]
    if not residue_h5_files:
        print(f"No H5 files found in directory: {input_residue_h5_dir}");
        return
    print(f"Found {len(residue_h5_files)} residue H5 files to process.")

    if num_workers is None: num_workers = max(1, cpu_count() - 1)

    # 4. Process each residue H5 file
    for h5_filename in residue_h5_files:
        current_residue_h5_path = os.path.join(input_residue_h5_dir, h5_filename)
        print(f"\nProcessing residue file: {h5_filename}...")

        residue_map, original_embedding_dim = load_residue_embeddings_from_h5(current_residue_h5_path)
        if residue_map is None:
            print(f"Skipping {h5_filename} due to loading error.")
            continue

        process_func_with_args = partial(process_single_protein,
                                         residue_embeddings_map=residue_map,
                                         unknown_residue_key='X')

        raw_protein_embeddings_results = []  # List of (uniprot_id, mean_embedding_vector)
        with Pool(processes=num_workers) as pool:
            results_iterator = pool.imap_unordered(process_func_with_args, protein_tasks)
            for result in tqdm(results_iterator, total=len(protein_tasks), desc=f"Proteins ({os.path.basename(h5_filename)})"):
                raw_protein_embeddings_results.append(result)

        # Filter out None embeddings and prepare for PCA
        uids_for_pca = [res[0] for res in raw_protein_embeddings_results if res[1] is not None]
        embeddings_for_pca = [res[1] for res in raw_protein_embeddings_results if res[1] is not None]

        if not embeddings_for_pca:
            print(f"No valid protein embeddings generated from {h5_filename}. Skipping PCA and save.")
            continue

        print(f"Generated {len(embeddings_for_pca)} raw protein embeddings from {h5_filename}.")

        # Apply PCA
        transformed_embeddings = apply_pca_to_embeddings(embeddings_for_pca, target_pca_dimension, original_embedding_dim)

        if transformed_embeddings is None:
            print(f"PCA failed or produced no result for {h5_filename}. Skipping save for this file.")
            continue

        if transformed_embeddings.shape[1] != target_pca_dimension:
            print(f"Warning: PCA output dimension ({transformed_embeddings.shape[1]}) for {h5_filename} "
                  f"does not strictly match target ({target_pca_dimension}) after PCA constraints. "
                  "This might happen if n_samples or original_dim < target_pca_dimension.")
            # Depending on strictness, one might choose to skip saving or pad here.
            # For now, we save what PCA returned if its dimension is at least 1.
            if transformed_embeddings.shape[1] == 0:
                print(f"PCA output dimension is 0 for {h5_filename}. Skipping save.");
                continue

        # Save PCA-transformed embeddings to a new H5 file
        output_filename_base = os.path.splitext(h5_filename)[0]
        output_h5_path = os.path.join(output_protein_h5_dir, f"{output_filename_base}_proteins_pca_dim{transformed_embeddings.shape[1]}.h5")

        successful_saves = 0
        with h5py.File(output_h5_path, 'w') as hf_out:
            for uid, transformed_vec in zip(uids_for_pca, transformed_embeddings):
                hf_out.create_dataset(str(uid), data=transformed_vec)
                successful_saves += 1

        print(f"Saved {successful_saves} PCA-transformed protein embeddings (dim={transformed_embeddings.shape[1]}) "
              f"to '{os.path.basename(output_h5_path)}'.")

    print("\n--- Batch Protein Embedding Generation Finished ---")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("===== Protein Embedding Generation Script (Batch Processing with PCA) =====")

    # --- Section 1: Automated Test Case Run ---
    print("\n--- Running Automated Test Case ---")
    # Test parameters
    data_directory = "C:/tmp/Models/dummy/"
    TEST_PROTEIN_CSV = os.path.join(data_directory,'test_proteins.csv')
    TEST_RESIDUE_H5_DIR = os.path.join(data_directory,'test_residue_h5_input_dir')
    TEST_PROTEIN_H5_OUTPUT_DIR = os.path.join(data_directory,'test_protein_h5_output_dir')
    TEST_NUM_PROTEINS = 10  # More proteins for PCA to be more meaningful
    TEST_TARGET_PCA_DIM = 3  # Target a small dimension for testing

    # Create dummy directories
    if os.path.exists(TEST_RESIDUE_H5_DIR): shutil.rmtree(TEST_RESIDUE_H5_DIR)
    os.makedirs(TEST_RESIDUE_H5_DIR)
    if os.path.exists(TEST_PROTEIN_H5_OUTPUT_DIR): shutil.rmtree(TEST_PROTEIN_H5_OUTPUT_DIR)
    # Output dir will be created by the main function if it doesn't exist

    # Create dummy protein CSV (with header for test simplicity)
    create_dummy_protein_csv(TEST_PROTEIN_CSV, num_proteins=TEST_NUM_PROTEINS, has_header=True,
                             id_col_name='protein_id', seq_col_name='protein_sequence')

    # Create multiple dummy residue H5 files with different initial dimensions
    create_dummy_residue_h5(os.path.join(TEST_RESIDUE_H5_DIR, 'residues_set1_dim5.h5'), embed_dim=5, vocab=DUMMY_RESIDUE_VOCAB, file_idx=1)
    create_dummy_residue_h5(os.path.join(TEST_RESIDUE_H5_DIR, 'residues_set2_dim8.h5'), embed_dim=8, vocab=DUMMY_RESIDUE_VOCAB, file_idx=2)
    create_dummy_residue_h5(os.path.join(TEST_RESIDUE_H5_DIR, 'residues_set3_dim4.h5'), embed_dim=4, vocab=DUMMY_RESIDUE_VOCAB, file_idx=3)  # Test case where original dim < target_pca_dim for some components
    create_dummy_residue_h5(os.path.join(TEST_RESIDUE_H5_DIR, 'residues_set4_bad.txt'), embed_dim=5, vocab=DUMMY_RESIDUE_VOCAB)  # Not an H5, should be skipped

    print("\nAutomated Test: Starting batch processing...")
    process_residue_embeddings_directory(
        protein_csv_path=TEST_PROTEIN_CSV,
        input_residue_h5_dir=TEST_RESIDUE_H5_DIR,
        output_protein_h5_dir=TEST_PROTEIN_H5_OUTPUT_DIR,
        target_pca_dimension=TEST_TARGET_PCA_DIM,
        uniprot_id_col='protein_id',  # Using names as dummy CSV has header
        sequence_col='protein_sequence',
        protein_csv_header=0,  # First row is header
        num_workers=2  # Limit workers for test
    )

    print("\nAutomated Test: Verification...")
    if os.path.exists(TEST_PROTEIN_H5_OUTPUT_DIR):
        output_files = [f for f in os.listdir(TEST_PROTEIN_H5_OUTPUT_DIR) if f.endswith('.h5')]
        print(f"Found {len(output_files)} output H5 files in '{TEST_PROTEIN_H5_OUTPUT_DIR}'. Expected ~3.")
        for outfile in output_files:
            try:
                with h5py.File(os.path.join(TEST_PROTEIN_H5_OUTPUT_DIR, outfile), 'r') as hf:
                    print(f"  Verifying '{outfile}': Contains {len(hf.keys())} protein embeddings.")
                    if len(hf.keys()) > 0:
                        first_key = list(hf.keys())[0]
                        embed_shape = hf[first_key][:].shape
                        print(f"    Embedding shape for '{first_key}': {embed_shape}")
                        if not (embed_shape[0] <= TEST_TARGET_PCA_DIM and embed_shape[0] > 0):  # Dimension can be < target if original dim or n_samples was smaller
                            print(f"    WARNING: Output dimension {embed_shape[0]} for {outfile} is not as expected (<= {TEST_TARGET_PCA_DIM} and >0). Check PCA logic and constraints.")
            except Exception as e:
                print(f"    Error verifying '{outfile}': {e}")
    else:
        print(f"Automated Test: Output directory '{TEST_PROTEIN_H5_OUTPUT_DIR}' not created.")

    print("\nAutomated Test: Cleaning up test files and directories...")
    if os.path.exists(TEST_PROTEIN_CSV): os.remove(TEST_PROTEIN_CSV)
    if os.path.exists(TEST_RESIDUE_H5_DIR): shutil.rmtree(TEST_RESIDUE_H5_DIR)
    if os.path.exists(TEST_PROTEIN_H5_OUTPUT_DIR): shutil.rmtree(TEST_PROTEIN_H5_OUTPUT_DIR)
    print("Automated test cleanup complete.")
    print("--- Automated Test Case Finished ---")

    # --- Section 2: Example for User's Normal Case Run ---
    print("\n\n--- Example: How to Run for Your Own Files (Normal Case) ---")
    print("This section is for illustration. Modify paths/parameters and uncomment to run.")

    # Define paths and parameters for YOUR actual data
    data_directory = "C:/tmp/Models/"
    my_actual_protein_csv = os.path.join(data_directory,'uniprot_swissprot_sequences.csv')
    my_input_residue_h5_directory = os.path.join(data_directory,'output_char_embeddings_GlobalCharGraph_RandInitFeat_v2/32')
    my_output_protein_h5_directory = os.path.join(data_directory,'embeddings_to_evaluate')

    # Define your desired final embedding dimension after PCA
    # This dimension should ideally be less than or equal to the smallest original embedding dimension
    # of your input residue files AND less than or equal to the number of proteins you have.
    my_target_pca_dimension = 64 # Example: aiming for 64-dimensional protein embeddings

    # Scenario A: Your protein CSV file HAS a header row
    # protein_id_column = 'ProteinAccession'
    # sequence_column = 'FullSequence'
    # csv_header_setting = 0 # Indicates first row is header

    # Scenario B: Your protein CSV file does NOT have a header row
    protein_id_column = 0 # First column
    sequence_column = 1 # Second column
    csv_header_setting = None

    print("\nExample Call (NOT RUNNING AUTOMATICALLY - modify and uncomment)")
    process_residue_embeddings_directory(
        protein_csv_path=my_actual_protein_csv,
        input_residue_h5_dir=my_input_residue_h5_directory,
        output_protein_h5_dir=my_output_protein_h5_directory,
        target_pca_dimension=my_target_pca_dimension,
        uniprot_id_col=protein_id_column,      # Use variable set in Scenario A or B
        sequence_col=sequence_column,         # Use variable set in Scenario A or B
        protein_csv_header=csv_header_setting, # Use variable set in Scenario A or B
        num_workers=max(1, cpu_count() - 8)   # Example: adjust as needed
    )

    print("\nTo run on your own files, edit this script: update paths in the 'Normal Case' section,")
    print("configure your CSV parameters (header, column IDs/indices), set your target PCA dimension,")
    print("and then uncomment the 'process_residue_embeddings_directory' call.")
    print("Ensure you have scikit-learn installed: pip install scikit-learn pandas numpy h5py tqdm")
    print("===== End of Script =====")
