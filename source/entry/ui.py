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
from typing import List

import pandas as pd

from configuration.config import Config
from source.utils.data.data_utils import DataUtils
from source.utils.data.fasta_utils import FastaUtils


class UIManager:
    """Handles all user interface and console interaction logic."""

    @staticmethod
    def prompt_to_continue(step_completed: str) -> bool:
        """Asks the user if they want to continue to the next pipeline step."""
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
        """Asks the user if they want to switch to the faster regex mapping mode."""
        if not sys.stdin.isatty():
            print("--- Non-interactive session detected. Using configured ID mapping mode. ---")
            return False # Default to the configured (slower) mode

        while True:
            try:
                prompt = (f"\n💡 Smart Mapping Suggestion: Your FASTA file appears to be {confidence_score:.1%} compatible with the fast 'regex' parser.\n"
                          f"   Would you like to use the fast 'regex' mode for this run instead of the slow 'file' mode? (y/n): ")
                response = input(prompt).lower().strip()
                if response in ['y', 'yes']: return True
                if response in ['n', 'no']: return False
                print("  Invalid input. Please enter 'y' or 'n'.")
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled by user. Proceeding with configured mode.")
                return False

    @staticmethod
    def get_fasta_files_to_process(config: Config, temp_dir: Path) -> List[Path]:
        """Handles logic for downsampling and selecting FASTA files."""
        # --- REFACTOR: The interactive prompt is removed. ---
        # The Config class now ensures that ORIGINAL_SEQUENCE_FILE_PATHS contains only the single,
        # user-specified FASTA file. This method now only needs to handle the downsampling logic.

        # Downsampling logic
        should_downsample = config.SEQUENCE_DOWNSAMPLE_FRACTION and 0 < config.SEQUENCE_DOWNSAMPLE_FRACTION < 1.0
        if should_downsample:
            DataUtils.print_header(f"Downsampling FASTA files ({config.SEQUENCE_DOWNSAMPLE_FRACTION:.1%})")
            random.seed(config.RANDOM_STATE)
            for original_path in config.ORIGINAL_SEQUENCE_FILE_PATHS:
                with open(original_path, 'r', encoding='utf-8', errors='ignore') as f:
                    total_sequences = sum(1 for line in f if line.startswith('>'))
                if total_sequences == 0:
                    print(f"  - WARNING: No sequences found in {original_path.name}. Skipping.")
                    continue
                sample_size = int(total_sequences * config.SEQUENCE_DOWNSAMPLE_FRACTION)
                print(f"  - Sampling {sample_size} of {total_sequences} sequences...")
                sequence_iterator = FastaUtils.parse_sequences(
                    [original_path],
                    perform_cleaning=config.PROTGRAM_CLEAN_FASTA_ON_PARSE,
                    min_len=config.PROTGRAM_FASTA_MIN_LEN,
                    max_len=config.PROTGRAM_FASTA_MAX_LEN,
                    alphabet_type=config.PROTGRAM_FASTA_ALPHABET
                )
                sampled_sequences = DataUtils.reservoir_sample(sequence_iterator, sample_size, config.RANDOM_STATE)
                temp_fasta_path = temp_dir / f"{original_path.stem}_sampled.fasta"
                with open(temp_fasta_path, "w") as f:
                    for seq_id, sequence in sampled_sequences:
                        f.write(f">{seq_id}\n{sequence}\n")
                files_to_process.append(temp_fasta_path)
        else:
            print(f"\nUsing specified FASTA file: {config.ORIGINAL_SEQUENCE_FILE_PATHS[0].name}")
            return config.ORIGINAL_SEQUENCE_FILE_PATHS.copy()

    @staticmethod
    def launch_mlflow_ui(config: Config):
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
                float_cols = formatted_group_df.select_dtypes(include=['float']).columns
                for col in float_cols:
                    formatted_group_df[col] = formatted_group_df[col].map('{:.4f}'.format)
                print(formatted_group_df[columns_to_print].to_string(index=False, na_rep='NaN'))
        except Exception as e:
            import traceback
            print(f"\n--- ❌ ERROR: Could not generate the aggregated benchmark summary. ---")
            print(f"This can happen if the results dataframes have an unexpected structure or contain invalid data.")
            print(f"Error details: {e}")
            traceback.print_exc()
