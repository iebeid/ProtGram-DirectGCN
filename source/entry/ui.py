# ==============================================================================
# MODULE: source/entry/ui.py
# PURPOSE: Manages all user interface interactions, including prompts and UI launches.
# VERSION: 2.0 (Consolidated)
# AUTHOR: Gemini Code Assist
# ==============================================================================

import os
import platform
import random
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import List, TYPE_CHECKING

import pandas as pd

from source.utils.data.data_utils import DataUtils
from source.utils.data.fasta_utils import FastaUtils

if TYPE_CHECKING:
    from configuration.config import Config


class UIManager:
    """Handles all user interface and console interaction logic."""

    @staticmethod
    def prompt_to_continue(step_completed: str, config: 'Config') -> bool:
        """Asks the user if they want to continue to the next pipeline step."""
        if config.DISABLE_INTERACTIVE_PROMPTS:
            print(f"--- Interactive prompts disabled. Automatically continuing after '{step_completed}'. ---")
            return True

        if not sys.stdin.isatty():
            print(f"--- Non-interactive session detected. Automatically continuing after '{step_completed}'. ---")
            return True

        while True:
            # Use a direct read from stdin to be more robust
            try:
                print(f"\n✅ Pipeline step '{step_completed}' is complete. Continue to the next step? (y/n): ", end="")
                sys.stdout.flush()
                response = sys.stdin.readline().strip().lower()
                if response in ['y', 'yes']:
                    return True
                if response in ['n', 'no']:
                    print("Exiting as requested by user.")
                    return False
                print("Invalid input. Please enter 'y' or 'n'.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user. Exiting.")
                return False

    @staticmethod
    def prompt_for_fast_id_mapping(confidence_score: float) -> bool:
        """Asks the user if they want to switch to the faster regex mapping mode, using a robust readline prompt."""
        if not sys.stdin.isatty():
            print("--- Non-interactive session detected. Using configured ID mapping mode. ---")
            return False # Default to the configured (slower) mode

        while True:
            try:
                # --- REFACTOR: Use direct stdin/stdout for consistency with other prompts ---
                prompt = (f"\n💡 Smart Mapping Suggestion: Your FASTA file appears to be {confidence_score:.1%} compatible with the fast 'regex' parser.\n" # noqa
                          f"   Would you like to use the fast 'regex' mode for this run instead of the slow 'file' mode? (y/n): ") # noqa
                sys.stdout.write(prompt)
                sys.stdout.flush()
                response = sys.stdin.readline().strip().lower()
                if response in ['y', 'yes']: return True
                if response in ['n', 'no']: return False
                print("  Invalid input. Please enter 'y' or 'n'.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user. Proceeding with configured mode.")
                return False

    @staticmethod
    def get_fasta_files_to_process(config: 'Config', temp_dir: Path) -> List[Path]:
        """Handles logic for downsampling and selecting FASTA files."""
        # --- REFACTOR: The interactive prompt is removed. ---
        # The Config class now ensures that ORIGINAL_SEQUENCE_FILE_PATHS contains only the single,
        # user-specified FASTA file. This method now only needs to handle the downsampling logic.

        should_downsample = config.SEQUENCE_DOWNSAMPLE_FRACTION and 0 < config.SEQUENCE_DOWNSAMPLE_FRACTION < 1.0
        if should_downsample:
            DataUtils.print_header(f"Downsampling FASTA files ({config.SEQUENCE_DOWNSAMPLE_FRACTION:.1%})")
            files_to_process = []
            for original_path in config.ORIGINAL_SEQUENCE_FILE_PATHS:
                # --- DEFINITIVE FIX: Use a single-pass probabilistic sampler for efficiency ---
                # The previous method read the file twice (once to count, once to sample).
                # This new method reads the file only once, yielding sequences with a given
                # probability, which is much faster for large files.
                print(f"  - Applying probabilistic sampling to '{original_path.name}'...")
                sequence_iterator = FastaUtils.parse_sequences(
                    [original_path],
                    perform_cleaning=config.PROTGRAM_CLEAN_FASTA_ON_PARSE,
                    min_len=config.PROTGRAM_FASTA_MIN_LEN,
                    max_len=config.PROTGRAM_FASTA_MAX_LEN,
                    alphabet_type=config.PROTGRAM_FASTA_ALPHABET
                )

                def probabilistic_sampler(iterator, fraction, seed):
                    rng = random.Random(seed)
                    for item in iterator:
                        if rng.random() < fraction:
                            yield item

                sampled_sequences = probabilistic_sampler(sequence_iterator, config.SEQUENCE_DOWNSAMPLE_FRACTION, config.RANDOM_STATE)
                temp_fasta_path = temp_dir / f"{original_path.stem}_sampled.fasta"
                with open(temp_fasta_path, "w") as f:
                    for seq_id, sequence in sampled_sequences:
                        f.write(f">{seq_id}\n{sequence}\n")
                files_to_process.append(temp_fasta_path)
            return files_to_process
        else:
            print(f"\nUsing specified FASTA file: {config.ORIGINAL_SEQUENCE_FILE_PATHS[0].name}")
            return config.ORIGINAL_SEQUENCE_FILE_PATHS.copy()

    @staticmethod
    def launch_mlflow_ui(config: 'Config'):
        """Starts the MLflow UI and opens a browser if in a desktop environment."""
        if not config.USE_MLFLOW:
            return

        DataUtils.print_header("Launching MLflow UI")
        tracking_uri = config.MLFLOW_TRACKING_URI
        is_desktop_env = os.environ.get('DISPLAY') or platform.system() == "Windows"

        if is_desktop_env:
            print("Desktop environment detected. Starting MLflow UI in the background...")
            subprocess.Popen(
                ["mlflow", "ui", "--backend-store-uri", str(tracking_uri)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(5)
            try:
                webbrowser.open("http://127.0.0.1:5000")
                print("\nMLflow UI has been launched in your web browser (or a new tab).")
                print("The MLflow server is running as a background process.")
                print("To stop it when you are finished, you may need to close this terminal or")
                print("manually find and stop the 'mlflow ui' process.")
            except webbrowser.Error as e:
                print(f"\nCould not automatically open web browser due to an error: {e}")
                print("Please open http://127.0.0.1:5000 manually to view results.")
        else:
            print("--- Headless/SSH environment detected. ---")
            print("To view the MLflow UI, run the following command on your local machine:")
            print(f"\n  mlflow ui --backend-store-uri {tracking_uri}\n")
            print("If running on a remote server, you may need to use SSH port forwarding, for example:")
            print("  ssh -L 5000:localhost:5000 your_user@your_server")

    @staticmethod
    @staticmethod
    def prompt_graph_objects_reuse_or_rebuild(config: 'Config', dataset_name: str, expected_output_dir: Path, n_max: int) -> dict:
        """
        Prompts the user to decide how to provide graph objects for the current run:
        - manual: User manually copies prebuilt graphs into expected_output_dir and confirms.
        - auto_copy: User provides a source directory and we copy graphs automatically.
        - rebuild: Run the graph builder to generate graphs from scratch.

        Returns a dict with keys: {'choice': 'manual'|'auto_copy'|'rebuild', 'source': Optional[Path]}
        """
        result = {'choice': 'rebuild', 'source': None}
        if config.DISABLE_INTERACTIVE_PROMPTS or not sys.stdin.isatty():
            # Non-interactive mode: default to rebuild
            print("--- Interactive prompts disabled or not a TTY. Proceeding to rebuild graph objects. ---")
            return result

        print("\n=== Graph Objects Availability ===")
        print(f"Target dataset: {dataset_name}")
        print(f"Expected graph objects directory:\n  {expected_output_dir}")
        print("\nYou have three options:")
        print("  1) Manually copy: I will copy previously built graph objects into the directory above and then continue.")
        print("  2) Auto-copy:     Copy graph objects automatically from a previous run directory I will provide.")
        print("  3) Rebuild:       Rebuild graph objects now (may take significant time).")

        while True:
            try:
                sys.stdout.write("\nChoose an option [1=manual copy, 2=auto-copy, 3=rebuild]: ")
                sys.stdout.flush()
                resp = sys.stdin.readline().strip()
                if resp in ('1', '2', '3'):
                    break
                print("  Invalid input. Please enter 1, 2, or 3.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user. Defaulting to rebuild.")
                return result

        if resp == '1':
            print("\nManual copy selected.")
            print(f"Please copy directories named 'ngram_graph_n1'..'ngram_graph_n{n_max}' into:")
            print(f"  {expected_output_dir}")
            print("Press Enter when you are done and ready to continue...")
            try:
                sys.stdin.readline()
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user. Defaulting to rebuild.")
                return result
            result['choice'] = 'manual'
            return result

        if resp == '2':
            print("\nAuto-copy selected.")
            print("Provide the absolute path to the directory containing your prebuilt graph objects.")
            print("You can specify either:")
            print("  - The dataset directory with 'ngram_graph_n*' subdirectories, or")
            print("  - The parent 'graph_objects' directory that contains one or more dataset subdirectories.")
            sys.stdout.write("Enter source path: ")
            sys.stdout.flush()
            try:
                src_str = sys.stdin.readline().strip()
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user. Defaulting to rebuild.")
                return result

            src_path = Path(src_str).expanduser().resolve()
            if not src_path.exists():
                print(f"  ❌ Source path not found: {src_path}")
                return result

            result['choice'] = 'auto_copy'
            result['source'] = src_path
            return result

        # Default to rebuild
        print("\nRebuild selected. Proceeding to rebuild graph objects.")
        result['choice'] = 'rebuild'
        return result

    @staticmethod
    def copy_graph_objects_from_source(source_root: Path, dest_dataset_dir: Path, n_max: int) -> bool:
        """
        Copies n-gram graph directories from source_root to dest_dataset_dir.
        Accepts either:
          - source_root containing 'ngram_graph_n*' directly, or
          - a parent directory containing a single dataset subdirectory.
        Returns True on success (all expected 'n' levels found and copied), False otherwise.
        """
        import shutil

        def has_ngram_dirs(p: Path) -> bool:
            return any((p / f"ngram_graph_n{i}").exists() for i in range(1, n_max + 1))

        src = source_root
        if not has_ngram_dirs(src):
            # Try to detect a single dataset subdirectory that has the ngram dirs
            candidates = [d for d in src.iterdir() if d.is_dir()]
            candidates = [d for d in candidates if has_ngram_dirs(d)]
            if len(candidates) == 1:
                src = candidates[0]
            else:
                print("  ❌ Could not locate 'ngram_graph_n*' directories under the provided path.")
                return False

        dest_dataset_dir.mkdir(parents=True, exist_ok=True)
        ok = True
        for n in range(1, n_max + 1):
            src_dir = src / f"ngram_graph_n{n}"
            dest_dir = dest_dataset_dir / f"ngram_graph_n{n}"
            if not src_dir.exists():
                print(f"  ❌ Missing source directory: {src_dir}")
                ok = False
                break
            # Remove existing dest to ensure a clean copy
            if dest_dir.exists():
                shutil.rmtree(dest_dir, ignore_errors=True)
            try:
                shutil.copytree(src_dir, dest_dir)
                print(f"  ✅ Copied {src_dir} -> {dest_dir}")
            except Exception as e:
                print(f"  ❌ Failed to copy {src_dir} -> {dest_dir}: {e}")
                ok = False
                break
        return ok

    @staticmethod
    def prompt_artifacts_reuse_or_copy(config: 'Config', dataset_name: str, expected_dir_map: dict) -> dict:
        """
        Prompts the user to decide how to provide ALL intermediate artifacts for the current run:
        - manual: User manually copies artifacts into the expected directories and then continues.
        - auto_copy: User provides a source directory and we copy artifacts automatically.
        - skip: Do nothing and proceed normally.

        expected_dir_map: {component_name (str) -> destination Path}
        Returns: {'choice': 'manual'|'auto_copy'|'skip', 'source': Optional[Path]}
        """
        result = {'choice': 'skip', 'source': None}
        # Non-interactive or prompts disabled: do nothing
        if config.DISABLE_INTERACTIVE_PROMPTS or not sys.stdin.isatty():
            return result

        print("\n=== Existing Artifacts Availability ===")
        print(f"Target dataset: {dataset_name}")
        print("You can populate the newly created results directories with artifacts from a previous run.")
        print("Components to consider:")
        for name, path in expected_dir_map.items():
            print(f"  - {name}: {path}")
        print("\nOptions:")
        print("  1) Manual copy: I will copy all desired artifacts into the listed directories and then continue.")
        print("  2) Auto-copy:   Copy artifacts automatically from a previous run directory I will provide.")
        print("  3) Skip:        Do nothing and proceed normally.")

        while True:
            try:
                sys.stdout.write("\nChoose an option [1=manual copy, 2=auto-copy, 3=skip]: ")
                sys.stdout.flush()
                resp = sys.stdin.readline().strip()
                if resp in ('1', '2', '3'):
                    break
                print("  Invalid input. Please enter 1, 2, or 3.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user. Defaulting to skip.")
                return result

        if resp == '1':
            print("\nManual copy selected.")
            print("Please copy the artifacts you want into the listed destination directories.")
            print("Press Enter when you are done and ready to continue...")
            try:
                sys.stdin.readline()
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user. Defaulting to skip.")
                return result
            result['choice'] = 'manual'
            return result

        if resp == '2':
            print("\nAuto-copy selected.")
            print("Provide the absolute path to a previous 'results' run directory or a dataset subdirectory under it.")
            print("We will try to locate and copy artifacts for the components listed above.")
            sys.stdout.write("Enter source path: ")
            sys.stdout.flush()
            try:
                src_str = sys.stdin.readline().strip()
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user. Defaulting to skip.")
                return result

            src_path = Path(src_str).expanduser().resolve()
            if not src_path.exists():
                print(f"  ❌ Source path not found: {src_path}")
                return result

            result['choice'] = 'auto_copy'
            result['source'] = src_path
            return result

        # Default: skip
        return result

    @staticmethod
    def copy_artifacts_from_source(source_root: Path, expected_dir_map: dict, n_max: int) -> dict:
        """
        Copies artifacts for all components from source_root into the destination directories in expected_dir_map.
        Heuristics:
          - If source_root/<component>/<dataset_name> exists, copy from there.
          - Else if source_root looks like a dataset-level directory (contains expected content), copy directly.
        Returns a dict: {component_name -> bool} indicating which components were successfully copied.
        """
        import shutil
        copied = {name: False for name in expected_dir_map.keys()}

        # Normalize component keys to their expected top-level directory names
        # expected_dir_map keys are already user-readable component names like:
        # 'graph_objects', 'gcn_embeddings', 'word2vec_embeddings', 'lstm_embeddings', 'transformer_embeddings', 'evaluation_results'
        def looks_like_graphs_dir(p: Path) -> bool:
            return any((p / f"ngram_graph_n{i}").exists() for i in range(1, n_max + 1))

        def looks_like_embeddings_dir(p: Path) -> bool:
            return any(p.glob("**/*.h5"))

        def looks_like_eval_dir(p: Path) -> bool:
            return (p / "evaluation_summary.csv").exists() or (p / "plots").exists()

        # Try component-wise copy
        for component, dest_dir in expected_dir_map.items():
            try:
                # First, try source_root/<component>/<dataset_name>
                # Detect if source_root is a 'results' root containing component directories
                potential_src = source_root / dest_dir.parent.name / dest_dir.name
                # Example: results/<component>/<dataset_name>
                if potential_src.exists():
                    # Copy entire directory tree
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    # Remove existing destination to ensure clean copy
                    if any(dest_dir.iterdir()):
                        shutil.rmtree(dest_dir, ignore_errors=True)
                        dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(potential_src, dest_dir, dirs_exist_ok=True)
                    print(f"  ✅ Copied {potential_src} -> {dest_dir}")
                    copied[component] = True
                    continue

                # Next, consider that source_root may itself be the dataset-level directory for this component
                # e.g., .../results/<component>/<dataset_name> (passed as source_root)
                # Use heuristics to decide if it matches the component type
                if looks_like_graphs_dir(source_root) and component == 'graph_objects':
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    if any(dest_dir.iterdir()):
                        shutil.rmtree(dest_dir, ignore_errors=True)
                        dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(source_root, dest_dir, dirs_exist_ok=True)
                    print(f"  ✅ Copied graphs {source_root} -> {dest_dir}")
                    copied[component] = True
                    continue

                if looks_like_embeddings_dir(source_root) and component.endswith('_embeddings'):
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    for item in source_root.glob("**/*.h5"):
                        rel = item.relative_to(source_root)
                        target = dest_dir / rel
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target)
                    print(f"  ✅ Copied embeddings {source_root} -> {dest_dir}")
                    copied[component] = True
                    continue

                if looks_like_eval_dir(source_root) and component == 'evaluation_results':
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    # Copy files and plots folder if present
                    for child in source_root.iterdir():
                        target = dest_dir / child.name
                        if child.is_dir():
                            if target.exists():
                                shutil.rmtree(target, ignore_errors=True)
                            shutil.copytree(child, target)
                        else:
                            shutil.copy2(child, target)
                    print(f"  ✅ Copied evaluation results {source_root} -> {dest_dir}")
                    copied[component] = True
                    continue

                # Lastly, try to detect a single dataset subdirectory under a component folder at source_root
                # For example: source_root/<component>/<some_dataset> where <some_dataset> contains expected structure.
                comp_parent = source_root / dest_dir.parent.name
                if comp_parent.exists() and comp_parent.is_dir():
                    candidates = [d for d in comp_parent.iterdir() if d.is_dir()]
                    # Prioritize dirs that "look like" relevant component
                    for d in candidates:
                        if component == 'graph_objects' and looks_like_graphs_dir(d):
                            src = d
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            if any(dest_dir.iterdir()):
                                shutil.rmtree(dest_dir, ignore_errors=True)
                                dest_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copytree(src, dest_dir, dirs_exist_ok=True)
                            print(f"  ✅ Copied graphs {src} -> {dest_dir}")
                            copied[component] = True
                            break
                        if component.endswith('_embeddings') and looks_like_embeddings_dir(d):
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            for item in d.glob("**/*.h5"):
                                rel = item.relative_to(d)
                                target = dest_dir / rel
                                target.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(item, target)
                            print(f"  ✅ Copied embeddings {d} -> {dest_dir}")
                            copied[component] = True
                            break
                        if component == 'evaluation_results' and looks_like_eval_dir(d):
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            for child in d.iterdir():
                                target = dest_dir / child.name
                                if child.is_dir():
                                    if target.exists():
                                        shutil.rmtree(target, ignore_errors=True)
                                    shutil.copytree(child, target)
                                else:
                                    shutil.copy2(child, target)
                            print(f"  ✅ Copied evaluation results {d} -> {dest_dir}")
                            copied[component] = True
                            break

            except Exception as e:
                print(f"  ❌ Failed to copy artifacts for component '{component}': {e}")
                copied[component] = False

        return copied

    @staticmethod
    def display_aggregated_benchmark_summary(all_results: List[pd.DataFrame]):
        """
        Standardizes, concatenates, and displays a final summary of all benchmark results.
        """
        if not all_results:
            print("No benchmark results were generated to aggregate.")
            return

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_colwidth', None)

        try:
            final_summary_df = pd.concat(all_results, ignore_index=True)
            all_columns = [
                'dataset', 'model', 'Accuracy', 'F1-Score (Macro)',
                'Precision (Macro)', 'Recall (Macro)', 'error'
            ]
            columns_to_print = [
                'model', 'Accuracy', 'F1-Score (Macro)',
                'Precision (Macro)', 'Recall (Macro)', 'error'
            ]
            final_summary_df = final_summary_df.reindex(columns=all_columns)

            def get_group_name(dataset_name):
                if not isinstance(dataset_name, str): return "Unknown"
                if 'ProtGram_n1_Singleton' in dataset_name:
                    return dataset_name
                return dataset_name.replace('_Original', '')

            model_order = [
                "GAT", "GraphSAGE", "GIN", "ChebNet", "Node2Vec", "GCN",
                "RGCN", "DirGNN", "DirectGCN"
            ]
            final_summary_df['dataset_group'] = final_summary_df['dataset'].apply(get_group_name)
            final_summary_df['model'] = pd.Categorical(final_summary_df['model'], categories=model_order, ordered=True)
            final_summary_df = final_summary_df.sort_values(by=['dataset_group', 'model'])

            DataUtils.print_header("Aggregated Benchmark Summary")
            for group_name, group_df in final_summary_df.groupby('dataset_group', sort=False):
                print(f"\n--- Results for Dataset: {group_name} ---")
                formatted_group_df = group_df.copy()
                # --- FIX: Make formatting robust to NaN and missing columns ---
                for col in columns_to_print:
                    if col not in formatted_group_df:
                        formatted_group_df[col] = 'N/A' # Add missing columns
                float_cols = formatted_group_df.select_dtypes(include=['float']).columns
                for col in float_cols:
                    formatted_group_df[col] = formatted_group_df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
                print(formatted_group_df[columns_to_print].to_string(index=False, na_rep='N/A'))
        except Exception as e:
            import traceback
            print(f"\n--- ❌ ERROR: Could not generate the aggregated benchmark summary. ---")
            print(f"This can happen if the results dataframes have an unexpected structure or contain invalid data.")
            print(f"Error details: {e}")
            traceback.print_exc()
