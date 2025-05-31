# run_ppi_evaluation.py
import os
import gc
import numpy as np
import tensorflow as tf
from typing import List, Optional, Dict, Any, Set, Tuple, Union

# Import from our new utility and CV worker scripts
from bioeval_utils import (
    ProteinFileOps, FileOps, Graph,
    plot_training_history, plot_roc_curves, plot_comparison_charts,
    print_results_table, perform_statistical_tests
)
from bioeval_cv_worker import main_workflow_cv

# --- USER CONFIGURATION SECTION ---
POSITIVE_INTERACTIONS_PATH = os.path.normpath('C:/tmp/Models/ground_truth/positive_interactions.csv')
NEGATIVE_INTERACTIONS_PATH = os.path.normpath('C:/tmp/Models/ground_truth/negative_interactions.csv')
SAMPLE_NEGATIVE_PAIRS: Optional[int] = 500000
DEFAULT_EMBEDDING_LOADER = 'load_h5_embeddings_selectively'
EMBEDDING_FILES_TO_COMPARE = [
    {
        "path": "C:/tmp/Models/embeddings_to_evaluate/pooled_proteins_from_CharEmb_ASCII_FIX_GlobalCharGraph_Directed_CustomGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2.h5",
        "name": "Directed_CustomGCN"
    },
    {
        "path": "C:/tmp/Models/embeddings_to_evaluate/pooled_proteins_from_CharEmb_ASCII_FIX_GlobalCharGraph_Directed_Tong_Library_DiGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2.h5",
        "name": "Directed_Tong"
    },{
        "path": "C:/tmp/Models/embeddings_to_evaluate/pooled_proteins_from_CharEmb_ASCII_FIX_GlobalCharGraph_Directed_UserCustomDiGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2.h5",
        "name": "Directed_UserCustomDiGCN"
    },
    {
        "path": "C:/tmp/Models/embeddings_to_evaluate/pooled_proteins_from_CharEmb_ASCII_FIX_GlobalCharGraph_Undirected_CustomGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2.h5",
        "name": "Undirected_CustomGCN"
    },{
        "path": "C:/tmp/Models/embeddings_to_evaluate/pooled_proteins_from_CharEmb_ASCII_FIX_GlobalCharGraph_Undirected_Tong_Library_DiGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2.h5",
        "name": "Undirected_Tong"
    },
    {
        "path": "C:/tmp/Models/embeddings_to_evaluate/pooled_proteins_from_CharEmb_ASCII_FIX_GlobalCharGraph_Undirected_UserCustomDiGCN_CHAR_EMBS_dim32_GlobalCharGraph_RandInitFeat_v2.h5",
        "name": "Undirected_UserCustomDiGCN"
    },
    {
        "path": "C:/tmp/Models/embeddings_to_evaluate/prot-t5-uniprot-per-residue.h5",
        "name": "ProtT5_Example_Data"
    }
]
# --- END OF USER CONFIGURATION SECTION ---

# --- SCRIPT BEHAVIOR AND MODEL PARAMETERS ---
DEBUG_VERBOSE = True
RANDOM_STATE = 42
EDGE_EMBEDDING_METHOD = 'concatenate'
N_FOLDS = 5
MAX_TRAIN_SAMPLES_CV = 10000
MAX_VAL_SAMPLES_CV = 5000
MAX_SHUFFLE_BUFFER_SIZE = 200000
PLOT_TRAINING_HISTORY = False  # Added missing variable definition

MLP_DENSE1_UNITS = 128
MLP_DROPOUT1_RATE = 0.4
MLP_DENSE2_UNITS = 64
MLP_DROPOUT2_RATE = 0.4
MLP_L2_REG = 0.001

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3

K_VALUES_FOR_RANKING_METRICS = [10, 50, 100, 200]
K_VALUES_FOR_TABLE_DISPLAY = [50, 100]

MAIN_EMBEDDING_NAME = "ProtT5_Example_Data"
STATISTICAL_TEST_METRIC_KEY = 'test_auc_sklearn'
STATISTICAL_TEST_ALPHA = 0.05

# This map should now point to functions in bioeval_utils.FileOps
LOADER_FUNCTION_MAP = {
    'load_h5_embeddings_selectively': lambda p, req_ids=None: FileOps.load_h5_embeddings_selectively(p,
                                                                                                     required_ids=req_ids),
    'load_h5_embeddings': lambda p, req_ids=None: FileOps.load_h5_embeddings_selectively(p, required_ids=req_ids),
    # Default old name to new
    'load_custom_embeddings': lambda p, req_ids=None: FileOps.load_custom_embeddings(p, required_ids=req_ids)
}

# Output directory for plots and results summary file
OUTPUT_DIR_BASE = "C:/tmp/Models/ppi_evaluation_results_v2/"
PLOTS_DIR = os.path.join(OUTPUT_DIR_BASE, "plots")
RESULTS_SUMMARY_FILE = os.path.join(OUTPUT_DIR_BASE,
                                    "ppi_evaluation_summary_table.txt")  # This var isn't directly used later, but good for reference

# --- Main Execution ---
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')

    # Set global DEBUG_VERBOSE for utils if needed.
    # This approach (setting module-level vars in an imported module) works but consider passing as params for clarity in larger projects.
    # Import bioeval_utils and set its global configurations
    import bioeval_utils

    bioeval_utils.DEBUG_VERBOSE = DEBUG_VERBOSE  #
    bioeval_utils.RANDOM_STATE = RANDOM_STATE  # This sets the global in bioeval_utils for other functions if they use it
    # ... (other global settings for bioeval_utils)

    print("Loading all interaction pairs ONCE to determine required proteins...")
    positive_pairs_all = ProteinFileOps.load_interaction_pairs(
        POSITIVE_INTERACTIONS_PATH, 1
    )  #
    negative_pairs_all = ProteinFileOps.load_interaction_pairs(
        NEGATIVE_INTERACTIONS_PATH, 0,
        sample_n=SAMPLE_NEGATIVE_PAIRS,  # Pass the sampling parameter
        random_state_for_sampling=RANDOM_STATE  # Pass the script's random state for reproducibility
    )  #

    all_interaction_pairs_for_ids = positive_pairs_all + negative_pairs_all  #
    if not all_interaction_pairs_for_ids:
        print("No interaction pairs loaded (possibly due to sampling all to zero). Exiting.")  #
        exit()

    required_protein_ids_for_interactions = set()  #
    for p1, p2, _ in all_interaction_pairs_for_ids:
        required_protein_ids_for_interactions.add(p1)  #
        required_protein_ids_for_interactions.add(p2)  #
    print(
        f"Found {len(required_protein_ids_for_interactions)} unique protein IDs in interaction files (after any sampling) that need embeddings.")  #

    # ... (rest of the script)
    # The `positive_pairs_all` and (now potentially sampled) `negative_pairs_all`
    # are then passed to main_workflow_cv as before.
    # cv_run_result = main_workflow_cv(
    #     ...
    #     positive_pairs=positive_pairs_all,
    #     negative_pairs=negative_pairs_all, # This will be the sampled list
    #     ...
    # )

    print(f"Numpy Version: {np.__version__}")
    print(f"Tensorflow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    gpu_devices = tf.config.list_physical_devices('GPU')
    print(f"GPU Devices: {gpu_devices}")
    if not gpu_devices:
        print("Warning: No GPU detected by TensorFlow.")

    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if not os.path.exists(POSITIVE_INTERACTIONS_PATH) or not os.path.exists(NEGATIVE_INTERACTIONS_PATH):
        print("CRITICAL ERROR: Interaction file paths are invalid. Exiting.")
        exit()

    EMBEDDING_CONFIGURATIONS_PROCESSED: List[Dict[str, Any]] = []
    for item in EMBEDDING_FILES_TO_COMPARE:
        config = {}
        path, name, loader_name_str = None, None, DEFAULT_EMBEDDING_LOADER
        if isinstance(item, str):
            path = item
        elif isinstance(item, dict):
            path = item.get('path')
            name = item.get('name')
            loader_name_str = item.get('loader', DEFAULT_EMBEDDING_LOADER)
        else:
            print(f"Warning: Invalid item in EMBEDDING_FILES_TO_COMPARE: {item}. Skipping.")
            continue

        if not path:
            print(f"Warning: Path missing in item: {item}. Skipping.")
            continue

        norm_path = os.path.normpath(path)
        if not os.path.exists(norm_path):
            print(f"Warning: Embedding path does not exist: {norm_path}. Skipping.")
            continue

        config['path'] = norm_path
        config['name'] = name if name else os.path.splitext(os.path.basename(norm_path))[0]

        actual_loader_func = LOADER_FUNCTION_MAP.get(loader_name_str)
        if not actual_loader_func:
            print(f"Warning: Loader '{loader_name_str}' not found for {config['name']}. Trying default loader.")
            actual_loader_func = LOADER_FUNCTION_MAP.get(DEFAULT_EMBEDDING_LOADER)
            if not actual_loader_func:
                print(
                    f"CRITICAL: Default loader '{DEFAULT_EMBEDDING_LOADER}' also not found. Skipping {config['name']}.")
                continue
        config['loader_func'] = actual_loader_func
        EMBEDDING_CONFIGURATIONS_PROCESSED.append(config)

    if not EMBEDDING_CONFIGURATIONS_PROCESSED:
        print("No valid embedding configurations found after checking paths and loaders. Exiting.")
        exit()

    all_cv_results: List[Dict[str, Any]] = []
    mlp_parameters = {
        'dense1_units': MLP_DENSE1_UNITS, 'dropout1_rate': MLP_DROPOUT1_RATE,
        'dense2_units': MLP_DENSE2_UNITS, 'dropout2_rate': MLP_DROPOUT2_RATE,
        'l2_reg': MLP_L2_REG
    }

    for config_item in EMBEDDING_CONFIGURATIONS_PROCESSED:
        print(f"\n{'=' * 25} Processing CV for: {config_item['name']} (Path: {config_item['path']}) {'=' * 25}")
        protein_embeddings = config_item['loader_func'](config_item['path'],
                                                        req_ids=required_protein_ids_for_interactions)

        if protein_embeddings and len(protein_embeddings) > 0:
            actually_loaded_ids = set(protein_embeddings.keys())
            relevant_loaded_ids = actually_loaded_ids.intersection(required_protein_ids_for_interactions)

            if not relevant_loaded_ids:
                print(
                    f"Skipping {config_item['name']}: No relevant protein embeddings loaded that are part of the interaction dataset.")
                all_cv_results.append({
                    'embedding_name': config_item['name'],
                    'notes': "No relevant embeddings found matching interaction data.",
                    **{k: 0.0 for k in ['test_f1_sklearn', 'test_auc_sklearn']},  # Default scores
                    'fold_f1_scores': [], 'fold_auc_scores': []
                })
                continue
            elif len(relevant_loaded_ids) < 2 and len(required_protein_ids_for_interactions) > 1:
                print(
                    f"Skipping {config_item['name']}: Insufficient relevant protein embeddings loaded ({len(relevant_loaded_ids)} found). At least 2 are needed for pair formation.")
                all_cv_results.append({
                    'embedding_name': config_item['name'],
                    'notes': f"Insufficient relevant embeddings: {len(relevant_loaded_ids)} IDs found.",
                    **{k: 0.0 for k in ['test_f1_sklearn', 'test_auc_sklearn']},  # Default scores
                    'fold_f1_scores': [], 'fold_auc_scores': []
                })
                continue

            cv_run_result = main_workflow_cv(
                embedding_name=config_item['name'],
                protein_embeddings=protein_embeddings,
                positive_pairs=positive_pairs_all,
                negative_pairs=negative_pairs_all,
                mlp_params=mlp_parameters,
                #plots_output_dir=PLOTS_DIR,
                edge_embedding_method=EDGE_EMBEDDING_METHOD,
                n_folds=N_FOLDS,
                random_state=RANDOM_STATE,
                max_train_samples_cv=MAX_TRAIN_SAMPLES_CV,
                max_val_samples_cv=MAX_VAL_SAMPLES_CV,
                max_shuffle_buffer_size=MAX_SHUFFLE_BUFFER_SIZE,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                k_values_for_ranking=K_VALUES_FOR_RANKING_METRICS
            )
            if cv_run_result:
                all_cv_results.append(cv_run_result)
                history_fold1 = cv_run_result.get('history_dict_fold1', {})
                if PLOT_TRAINING_HISTORY and history_fold1 and any(
                        isinstance(val_list, list) and len(val_list) > 0 for val_list in history_fold1.values()):
                    plot_training_history(history_fold1, cv_run_result['embedding_name'], PLOTS_DIR, fold_num=1)
                elif DEBUG_VERBOSE and PLOT_TRAINING_HISTORY:
                    print(
                        f"No/empty training history from fold 1 for {cv_run_result.get('embedding_name', 'Unknown Run')} to plot.")
        else:
            print(
                f"Skipping CV for {config_item['name']}: Failed to load or embeddings are empty after selective loading.")
            all_cv_results.append({
                'embedding_name': config_item['name'],
                'notes': "Failed to load or no relevant embeddings found.",
                **{k: 0.0 for k in ['test_f1_sklearn', 'test_auc_sklearn']},  # Default scores
                'fold_f1_scores': [], 'fold_auc_scores': []
            })

        del protein_embeddings  # Free up memory
        gc.collect()

    if all_cv_results:
        # The bioeval_utils globals were set at the beginning. No need to reset unless they changed,
        # but it's harmless here.
        bioeval_utils.K_VALUES_FOR_TABLE_DISPLAY = K_VALUES_FOR_TABLE_DISPLAY
        bioeval_utils.DEBUG_VERBOSE = DEBUG_VERBOSE  # Ensure it's current if it was changed
        bioeval_utils.STATISTICAL_TEST_ALPHA = STATISTICAL_TEST_ALPHA

        if DEBUG_VERBOSE:
            print("\nDEBUG (__main__): Final all_cv_results before aggregate plots/table (summary):")
            for i, res_dict in enumerate(all_cv_results):
                print(
                    f"  Summary for CV run {i + 1} ({res_dict.get('embedding_name')}): "
                    f"F1_avg={res_dict.get('test_f1_sklearn', 0.0):.4f}, "  # Added default for safety
                    f"AUC_avg={res_dict.get('test_auc_sklearn', 0.0):.4f}, "  # Added default for safety
                    f"Notes: {res_dict.get('notes')}"
                )

        print("\n\nGenerating aggregate comparison plots & table (based on CV averages)...")

        valid_roc_data_exists = any(
            res.get('roc_data_representative') and
            res['roc_data_representative'][0] is not None and
            hasattr(res['roc_data_representative'][0], '__len__') and
            len(res['roc_data_representative'][0]) > 0
            for res in all_cv_results if
            isinstance(res.get('roc_data_representative'), tuple) and len(res.get('roc_data_representative')) == 3
            # Ensure roc_data is tuple (fpr, tpr, auc)
        )
        if valid_roc_data_exists:
            plot_roc_curves(all_cv_results, PLOTS_DIR)
        else:
            print(
                "No valid representative ROC data to plot across models (e.g., all folds might have had only one class or data was missing).")

        plot_comparison_charts(all_cv_results, PLOTS_DIR, K_VALUES_FOR_TABLE_DISPLAY)
        print_results_table(all_cv_results, K_VALUES_FOR_TABLE_DISPLAY, is_cv=True, output_dir=OUTPUT_DIR_BASE,
                            filename="ppi_evaluation_summary_table.txt")

        perform_statistical_tests(all_cv_results, MAIN_EMBEDDING_NAME,
                                  metric_key_cfg=STATISTICAL_TEST_METRIC_KEY,
                                  alpha_cfg=STATISTICAL_TEST_ALPHA)
    else:
        print("\nNo results generated from any configurations to plot or tabulate.")

    print("\nScript finished.")