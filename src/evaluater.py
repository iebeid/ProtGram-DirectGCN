from importer import *

# --- USER CONFIGURATION SECTION ---
# Please update the paths and lists in this section carefully.

# 1. Specify Paths to Interaction Files:
#    These files define the known positive and negative protein-protein interactions.
POSITIVE_INTERACTIONS_PATH = 'G:/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/positive_interactions.csv'  # UPDATE THIS
NEGATIVE_INTERACTIONS_PATH = 'G:/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/negative_interactions.csv'  # UPDATE THIS

# 2. Define the Default Loader for Embedding Files:
#    This function will be used if a specific loader isn't provided for an embedding file.
#    load_h5_embeddings is for HDF5 files where keys are protein IDs and values are embeddings.
#    load_custom_embeddings is a placeholder for other formats (e.g., .npz as per its internal logic).
DEFAULT_EMBEDDING_LOADER = 'load_h5_embeddings'  # Can be 'load_h5_embeddings' or 'load_custom_embeddings'

# 3. List Your Embedding Files for Comparison:
#    Add paths to your embedding files here.
#    - For simple HDF5 files (compatible with load_h5_embeddings):
#      Just add the path as a string, e.g., "path/to/your/embeddings1.h5"
#      The name will be derived from the filename, and DEFAULT_EMBEDDING_LOADER will be used.
#
#    - For files requiring a custom name or a specific loader (e.g., NPZ files, custom formats):
#      Add a dictionary:
#      {
#          "path": "path/to/your/embeddings.npz",  # Mandatory
#          "name": "My NPZ Embeddings",            # Optional (derived from filename if not given)
#          "loader": "load_custom_embeddings"      # Optional (DEFAULT_EMBEDDING_LOADER if not given)
#      }
#
#    MAKE SURE TO UPDATE THE EXAMPLE PATHS BELOW.
EMBEDDING_FILES_TO_COMPARE = [
    "G:/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/per-protein.h5",  # Example: A standard HDF5 embedding file
    {
        "path": "G:/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/per-protein.h5",
        "name": "UniProt"
    },
    {
        "path": "G:/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/per-protein.h5",
        "name": "UniProt2"  # Uses DEFAULT_EMBEDDING_LOADER by default
    },
    # Add more of your embedding file paths or dictionaries here:
    # "path/to/your/yet_another_embeddings.h5",
]
# --- END OF USER CONFIGURATION SECTION ---


# Model and Training Parameters (generally less frequently changed)
EDGE_EMBEDDING_METHOD = 'concatenate'  # Options: 'concatenate', 'average', 'hadamard', 'subtract'
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

# Loader functions mapping string names to actual functions
LOADER_FUNCTION_MAP = {
    'load_h5_embeddings': lambda p: load_h5_embeddings(p),  # Defined below
    'load_custom_embeddings': lambda p: load_custom_embeddings(p)  # Defined below
}


# --- Helper Functions ---

def load_h5_embeddings(file_path: str) -> dict[str, np.ndarray]:
    """
    Loads protein embeddings from an HDF5 file.
    Assumes HDF5 structure: keys are protein IDs, values are embedding vectors.
    """
    print(f"Loading embeddings using 'load_h5_embeddings' from: {file_path}...")
    embeddings: dict[str, np.ndarray] = {}
    if not os.path.exists(file_path):
        print(f"Error: Embedding file not found at {file_path}")
        return embeddings
    try:
        with h5py.File(file_path, 'r') as hf:
            for protein_id in hf.keys():
                embeddings[protein_id] = np.array(hf[protein_id][:], dtype=np.float32)
        print(f"Loaded {len(embeddings)} embeddings from {file_path}.")
    except Exception as e:
        print(f"Error loading HDF5 embeddings from {file_path}: {e}")
    return embeddings


def load_custom_embeddings(file_path: str) -> dict[str, np.ndarray]:
    """
    Loads custom protein embeddings. Adapts based on file extension or internal logic.
    Modify this to suit the format of your embeddings file.
    Expected output: a dictionary {protein_id: embedding_vector (np.ndarray)}
    """
    print(f"Attempting to load embeddings using 'load_custom_embeddings' from: {file_path}...")
    embeddings: dict[str, np.ndarray] = {}
    if not os.path.exists(file_path):
        print(f"Error: Custom embedding file not found at {file_path}")
        return embeddings

    if file_path.endswith('.npz'):
        print(f"Detected .npz extension, attempting NPZ loading logic for {file_path}...")
        try:
            data = np.load(file_path, allow_pickle=True)
            if 'embeddings_dict' in data:
                loaded_item = data['embeddings_dict'].item()
                if isinstance(loaded_item, dict):
                    embeddings = {str(k): np.array(v, dtype=np.float32) for k, v in loaded_item.items()}
                else:
                    print(f"NPZ 'embeddings_dict' in {file_path} did not contain a dictionary.")
            elif 'ids' in data and 'vectors' in data:
                ids = data['ids']
                vectors = data['vectors']
                embeddings = {str(pid): vec.astype(np.float32) for pid, vec in zip(ids, vectors)}
            else:
                # Fallback for simple dict stored as the first item in npz
                if data.files:
                    loaded_obj = data[data.files[0]]
                    if isinstance(loaded_obj.item(), dict):
                        embeddings = {str(k): np.array(v, dtype=np.float32) for k, v in loaded_obj.item().items()}
                    else:
                        print(f"Could not interpret NPZ file structure in {file_path}. Expected 'embeddings_dict' or 'ids'/'vectors' keys, or a single dict item.")
                else:
                    print(f"NPZ file {file_path} is empty or has no recognizable data structure.")
            print(f"Loaded {len(embeddings)} custom embeddings from NPZ {file_path}.")
        except Exception as e:
            print(f"Error loading custom NPZ embeddings from {file_path}: {e}")
        return embeddings
    elif file_path.endswith('.h5'):  # Delegate to HDF5 loader if it's an H5 file
        print(f"Detected .h5 extension for {file_path}, delegating to 'load_h5_embeddings'.")
        return load_h5_embeddings(file_path)
    else:
        print(f"Warning: 'load_custom_embeddings' does not have specific logic for file type of {file_path}. "
              "Please adapt this function or ensure the file is an .npz or .h5 handled by default.")
        # TODO: Add your custom loading logic here for other formats if needed
        return {}


def load_interaction_pairs(file_path: str, label: int) -> list[tuple[str, str, int]]:
    print(f"Loading interaction pairs from: {file_path} (label: {label})...")
    pairs: list[tuple[str, str, int]] = []
    skipped_lines = 0
    if not os.path.exists(file_path):
        print(f"Error: Interaction file not found at {file_path}")
        return pairs
    try:
        # Explicitly set encoding, good practice.
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                current_line_stripped = line.strip()

                # Header check logic:
                if i == 0:
                    # Ensure we are operating on a string for these checks
                    if isinstance(current_line_stripped, str) and \
                            ('protein' in current_line_stripped.lower() or \
                             'interactor' in current_line_stripped.lower() or \
                             current_line_stripped.startswith('#')):
                        print(f"Skipping header line: {current_line_stripped}")
                        continue

                # Skip empty lines after header check
                if not current_line_stripped:
                    # It's an empty line, not necessarily malformed if it's not the header
                    # but we can choose to count it or not. Let's not count it as skipped for now
                    # unless it causes issues in splitting.
                    continue

                # Proceed with processing the line, using current_line_stripped
                parts = current_line_stripped.split('\t')
                if len(parts) < 2:
                    parts = current_line_stripped.split(',')  # Try comma if tab split fails

                if len(parts) >= 2:
                    # Strip individual protein IDs in case of leading/trailing whitespace
                    p1 = parts[0].strip()
                    p2 = parts[1].strip()
                    if p1 and p2:  # Ensure both protein IDs are non-empty after stripping
                        pairs.append((p1, p2, label))
                    else:
                        # One of the protein IDs became empty after stripping
                        if current_line_stripped:  # Only count if original line wasn't just whitespace
                            skipped_lines += 1
                elif current_line_stripped:  # Line was not empty but couldn't be split into at least 2 parts
                    skipped_lines += 1

    except Exception as e:
        print(f"CRITICAL Error during processing of {file_path}: {e}")
        import traceback
        print("Detailed traceback of the error:")
        traceback.print_exc()  # This will print the exact line where the error occurred
        # The function will return the pairs loaded so far, or an empty list if error was early.

    if skipped_lines > 0:
        print(f"Note: Skipped {skipped_lines} malformed or empty-ID lines in {file_path}.")
    # Moved this message to be outside the try-except, so it always reports what was loaded.
    print(f"Successfully loaded {len(pairs)} pairs from {file_path}.")
    return pairs


def create_edge_embeddings(
        interaction_pairs: list[tuple[str, str, int]],
        protein_embeddings: dict[str, np.ndarray],
        method: str = 'concatenate'
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    print(f"Creating edge embeddings using method: {method}...")
    edge_features_list = []
    labels_list = []
    skipped_pairs_missing_embeddings = 0
    skipped_pairs_mismatched_dim = 0

    if not protein_embeddings:
        print("Warning: Protein embeddings dictionary is empty. Cannot create edge features.")
        return None, None

    embedding_dim = 0
    for pid in protein_embeddings:  # Find first valid embedding to determine dimension
        if isinstance(protein_embeddings[pid], np.ndarray) and protein_embeddings[pid].ndim > 0:  # ensure it's an array and has a shape
            if protein_embeddings[pid].shape[0] > 0:  # ensure dimension is not zero
                embedding_dim = protein_embeddings[pid].shape[0]
                break

    if embedding_dim == 0:
        print("Warning: Could not determine a valid embedding dimension from provided protein embeddings (all embeddings might be empty or not found).")
        return None, None
    print(f"Inferred embedding dimension: {embedding_dim}")

    for p1_id, p2_id, label in interaction_pairs:
        emb1 = protein_embeddings.get(p1_id)
        emb2 = protein_embeddings.get(p2_id)

        if emb1 is not None and emb2 is not None:
            if emb1.shape[0] != embedding_dim or emb2.shape[0] != embedding_dim:
                skipped_pairs_mismatched_dim += 1
                continue

            if method == 'concatenate':
                edge_emb = np.concatenate((emb1, emb2))
            elif method == 'average':
                edge_emb = (emb1 + emb2) / 2.0
            elif method == 'hadamard':
                edge_emb = emb1 * emb2
            elif method == 'subtract':
                edge_emb = np.abs(emb1 - emb2)
            else:
                raise ValueError(f"Unknown edge embedding method: {method}")
            edge_features_list.append(edge_emb)
            labels_list.append(label)
        else:
            skipped_pairs_missing_embeddings += 1

    print(f"Created {len(edge_features_list)} edge features.")
    if skipped_pairs_missing_embeddings > 0:
        print(f"Skipped {skipped_pairs_missing_embeddings} pairs due to one or both protein embeddings not found.")
    if skipped_pairs_mismatched_dim > 0:
        print(f"Skipped {skipped_pairs_mismatched_dim} pairs due to mismatched embedding dimensions.")

    if not edge_features_list:
        print("No edge features were created. Check protein IDs, embedding files, and dimension consistency.")
        return None, None

    return np.array(edge_features_list, dtype=np.float32), np.array(labels_list, dtype=np.int32)


def build_mlp_model(input_shape: int, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc_keras'),
            tf.keras.metrics.Precision(name='precision_keras'),
            tf.keras.metrics.Recall(name='recall_keras')
        ]
    )
    return model


# --- Plotting Functions ---
def plot_training_history(history_dict: dict[str, Any], model_name: str):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Training Loss')
    if 'val_loss' in history_dict: plt.plot(history_dict['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss: {model_name}');
    plt.ylabel('Loss');
    plt.xlabel('Epoch');
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history_dict: plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy: {model_name}');
    plt.ylabel('Accuracy');
    plt.xlabel('Epoch');
    plt.legend()
    plt.suptitle(f"Training History: {model_name}", fontsize=16);
    plt.tight_layout(rect=[0, 0, 1, 0.96]);
    plt.show()


def plot_roc_curves(results_list: list[dict[str, Any]]):
    plt.figure(figsize=(10, 8))
    for result in results_list:
        if result and 'roc_data' in result and result['roc_data']:
            fpr, tpr, roc_auc_value = result['roc_data']
            plt.plot(fpr, tpr, lw=2, label=f"{result['embedding_name']} (AUC = {roc_auc_value:.3f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xlabel('FPR');
    plt.ylabel('TPR')
    plt.title('ROC Curves Comparison');
    plt.legend(loc="lower right");
    plt.grid(True);
    plt.show()


def plot_comparison_charts(results_list: list[dict[str, Any]]):
    if not results_list: print("No results to plot for comparison."); return
    metrics_to_plot = {'Accuracy': 'test_accuracy_keras', 'Precision': 'test_precision_sklearn',
                       'Recall': 'test_recall_sklearn', 'F1-Score': 'test_f1_sklearn', 'AUC': 'test_auc_sklearn'}
    embedding_names = [res['embedding_name'] for res in results_list]
    num_metrics, num_embeddings = len(metrics_to_plot), len(embedding_names)
    plt.figure(figsize=(max(15, num_embeddings * 2.5), 12))
    for i, (disp_name, key) in enumerate(metrics_to_plot.items()):
        values = [res.get(key, 0.0) for res in results_list]
        plt.subplot(2, (num_metrics + 1) // 2, i + 1)
        bars = plt.bar(embedding_names, values, color=plt.cm.viridis(np.linspace(0, 1, num_embeddings)))
        plt.title(disp_name);
        plt.ylabel('Score');
        plt.xticks(rotation=45, ha="right");
        plt.ylim(0, 1.05)
        for bar in bars: plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', va='bottom')
    training_times = [res.get('training_time', 0.0) for res in results_list]
    plt.subplot(2, (num_metrics + 1) // 2, num_metrics + 1)
    max_time = max(training_times) if training_times else 1
    bars = plt.bar(embedding_names, training_times, color=plt.cm.plasma(np.linspace(0, 1, num_embeddings)))
    plt.title('Training Time');
    plt.ylabel('Seconds');
    plt.xticks(rotation=45, ha="right");
    plt.ylim(0, max_time * 1.15 if max_time > 0 else 10)
    for bar in bars: plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01 * max_time, f'{bar.get_height():.2f}s', ha='center', va='bottom')
    plt.suptitle("Model Performance Comparison", fontsize=18);
    plt.tight_layout(rect=[0, 0, 1, 0.95]);
    plt.show()


# --- Main Workflow ---
def main_workflow(embedding_name: str, protein_embeddings: dict[str, np.ndarray]) -> Optional[dict[str, Any]]:
    results: dict[str, Any] = {'embedding_name': embedding_name}
    if not protein_embeddings: print(f"No embeddings for {embedding_name}. Skipping."); return None
    print(f"\n===== Starting Link Prediction for {embedding_name} =====")
    positive_pairs = load_interaction_pairs(POSITIVE_INTERACTIONS_PATH, 1)
    negative_pairs = load_interaction_pairs(NEGATIVE_INTERACTIONS_PATH, 0)
    if not positive_pairs and not negative_pairs: print(f"No interaction pairs for {embedding_name}."); return None
    all_interaction_pairs = positive_pairs + negative_pairs
    if not all_interaction_pairs: print(f"No combined interactions for {embedding_name}."); return None
    random.shuffle(all_interaction_pairs)
    X, y = create_edge_embeddings(all_interaction_pairs, protein_embeddings, method=EDGE_EMBEDDING_METHOD)
    if X is None or y is None or len(X) == 0: print(f"Dataset creation failed for {embedding_name}."); return None
    print(f"Total samples for {embedding_name}: {len(y)} (+:{np.sum(y == 1)}, -:{np.sum(y == 0)})")
    if len(np.unique(y)) < 2:
        print(f"Warning: Only one class in dataset for {embedding_name}. Cannot train/evaluate effectively.")
        results.update({'training_time': 0, 'history_dict': {}, 'roc_data': None, 'notes': "Single class data",
                        **{k: -1 for k in ['test_loss', 'test_accuracy_keras', 'test_auc_keras',
                                           'test_precision_keras', 'test_recall_keras', 'test_precision_sklearn',
                                           'test_recall_sklearn', 'test_f1_sklearn', 'test_auc_sklearn']}})
        return results
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y)
    except ValueError as e:
        print(f"Stratified split failed ({e}) for {embedding_name}. Using non-stratified.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE)
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    model = build_mlp_model(X_train.shape[1], LEARNING_RATE);
    model.summary(print_fn=lambda x: print(x))
    print(f"Training MLP for {embedding_name}...");
    start_time = time.time()
    history = model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, verbose=2)
    results['training_time'] = time.time() - start_time;
    results['history_dict'] = history.history
    print(f"Training for {embedding_name} took {results['training_time']:.2f}s.")
    print(f"Evaluating {embedding_name} (Keras)...");
    eval_res = model.evaluate(test_ds, verbose=0)
    keras_metrics = ['test_loss', 'test_accuracy_keras', 'test_auc_keras', 'test_precision_keras', 'test_recall_keras']
    for name, val in zip(keras_metrics, eval_res): results[name] = val; print(f"  Keras {name.split('_')[1]}: {val:.4f}")
    print(f"Generating predictions for {embedding_name} (Sklearn)...")
    y_pred_proba = model.predict(X_test, batch_size=BATCH_SIZE).flatten();
    y_pred_class = (y_pred_proba > 0.5).astype(int)
    results['test_precision_sklearn'] = precision_score(y_test, y_pred_class, zero_division=0)
    results['test_recall_sklearn'] = recall_score(y_test, y_pred_class, zero_division=0)
    results['test_f1_sklearn'] = f1_score(y_test, y_pred_class, zero_division=0)
    print(f"  Sklearn Precision: {results['test_precision_sklearn']:.4f}, Recall: {results['test_recall_sklearn']:.4f}, F1: {results['test_f1_sklearn']:.4f}")
    if len(np.unique(y_test)) > 1:
        results['test_auc_sklearn'] = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba);
        results['roc_data'] = (fpr, tpr, results['test_auc_sklearn'])
        print(f"  Sklearn AUC: {results['test_auc_sklearn']:.4f}")
    else:
        results['test_auc_sklearn'] = 0.0;
        results['roc_data'] = None;
        print("  Sklearn AUC: Not calculable (single class)")
    print(f"Confusion Matrix for {embedding_name}:\n{confusion_matrix(y_test, y_pred_class)}")
    del X, y, X_train, X_test, y_train, y_test, model, history, train_ds, test_ds;
    gc.collect();
    tf.keras.backend.clear_session()
    print(f"===== Finished Workflow for {embedding_name} =====");
    return results


# --- Main Execution ---
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2';
    tf.get_logger().setLevel('ERROR')

    # Preliminary check for interaction files
    if not os.path.exists(POSITIVE_INTERACTIONS_PATH) or not os.path.exists(NEGATIVE_INTERACTIONS_PATH):
        print("CRITICAL ERROR: Positive or Negative interaction file paths are invalid.")
        print(f"  Positive Path Check: {POSITIVE_INTERACTIONS_PATH} (Exists: {os.path.exists(POSITIVE_INTERACTIONS_PATH)})")
        print(f"  Negative Path Check: {NEGATIVE_INTERACTIONS_PATH} (Exists: {os.path.exists(NEGATIVE_INTERACTIONS_PATH)})")
        print("Please correct these paths in the USER CONFIGURATION SECTION. Exiting.")
        exit()

    # Process EMBEDDING_FILES_TO_COMPARE to build dynamic configurations
    EMBEDDING_CONFIGURATIONS = []
    for item in EMBEDDING_FILES_TO_COMPARE:
        config = {}
        path, name, loader_name_str = None, None, DEFAULT_EMBEDDING_LOADER  # Default loader name as string
        if isinstance(item, str):  # Item is just a path string
            path = item
        elif isinstance(item, dict):  # Item is a dictionary
            path = item.get('path')
            name = item.get('name')
            loader_name_str = item.get('loader', DEFAULT_EMBEDDING_LOADER)
        else:
            print(f"Warning: Invalid item type in EMBEDDING_FILES_TO_COMPARE. Skipping: {item}")
            continue

        if not path:
            print(f"Warning: Path missing for item in EMBEDDING_FILES_TO_COMPARE. Skipping: {item}")
            continue
        if not os.path.exists(path):
            print(f"Warning: Configured embedding path does not exist: {path}. Skipping.")
            continue

        config['path'] = path
        config['name'] = name if name else os.path.splitext(os.path.basename(path))[0]

        if loader_name_str not in LOADER_FUNCTION_MAP:
            print(f"Warning: Loader '{loader_name_str}' for {config['name']} not found in LOADER_FUNCTION_MAP. Using default.")
            loader_name_str = DEFAULT_EMBEDDING_LOADER  # Fallback to default if specified loader is unknown
        config['loader_func'] = LOADER_FUNCTION_MAP[loader_name_str]  # Store the actual function

        EMBEDDING_CONFIGURATIONS.append(config)

    if not EMBEDDING_CONFIGURATIONS:
        print("No valid embedding configurations were processed. Please check EMBEDDING_FILES_TO_COMPARE. Exiting.")
        exit()

    all_run_results: list[dict[str, Any]] = []
    for config_item in EMBEDDING_CONFIGURATIONS:
        print(f"\n{'=' * 25} Processing: {config_item['name']} (Path: {config_item['path']}) {'=' * 25}")
        protein_embeddings = config_item['loader_func'](config_item['path'])  # Call the resolved loader function

        if protein_embeddings and len(protein_embeddings) > 0:
            run_result = main_workflow(config_item['name'], protein_embeddings)
            if run_result:
                all_run_results.append(run_result)
                if run_result.get('history_dict') and run_result['history_dict']:
                    plot_training_history(run_result['history_dict'], run_result['embedding_name'])
                else:
                    print(f"No training history to plot for {run_result['embedding_name']}.")
        else:
            print(f"Skipping workflow for {config_item['name']}: Failed to load/empty embeddings.")
        del protein_embeddings;
        gc.collect()

    if all_run_results:
        print("\n\nGenerating aggregate comparison plots...");
        valid_roc_res = [r for r in all_run_results if r.get('roc_data')];
        if valid_roc_res:
            plot_roc_curves(valid_roc_res)
        else:
            print("No valid ROC data to plot across models.")
        plot_comparison_charts(all_run_results)
    else:
        print("\nNo results generated from any configurations to plot.")
    print("\nScript finished.")
