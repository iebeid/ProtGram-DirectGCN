# ==============================================================================
# MODULE: benchmarkers/base.py
# PURPOSE: A base class for benchmarking pipelines to reduce code duplication.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.datasets import Planetoid, WebKB, Actor, KarateClub

from configuration.config import Config
from typing import Optional, Any, List
from source.utils.data.data_utils import DataUtils


class BaseBenchmarker(ABC):
    """
    An abstract base class for benchmarking pipelines.
    Handles common logic like dataset loading, directory setup, and seeding.
    """

    def __init__(self, config: Config, name: str):
        self.config = config
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = config.RESULTS_BENCHMARKING_DIR
        self.dataset_root = self._setup_dataset_cache(config)
        DataUtils.set_seeds(config.RANDOM_STATE)
        print("\n" + "=" * 80)
        DataUtils.print_header(f"{self.name} Initialized")
        print(f"  Device: {self.device}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Dataset root: {self.dataset_root}")
        print("=" * 80)

    def _create_link_or_copy_dir(self, source: Path, dest: Path):
        """Creates a symlink for a directory, falling back to a copy if needed."""
        try:
            # --- REFACTOR: Use pathlib's symlink_to for better cross-platform compatibility ---
            # This is generally more robust than os.symlink, especially on Windows.
            dest.symlink_to(source, target_is_directory=True)
        except (OSError, AttributeError, NotImplementedError):
            print(f"    Symlink failed. Falling back to copying directory (this may take a moment)...")
            shutil.copytree(source, dest)

    def _setup_dataset_cache(self, config: Config) -> str:
        """Sets up a persistent cache for benchmark datasets and returns the path."""
        project_benchmark_dir = config.DATA_STANDARD_DATASETS_DIR
        cache_benchmark_dir = config.PERSISTENT_DATA_CACHE / "benchmarks"
        cache_benchmark_dir.mkdir(parents=True, exist_ok=True)

        if project_benchmark_dir.exists() and not project_benchmark_dir.is_symlink():
            print(f"  Migrating existing benchmark data from '{project_benchmark_dir.relative_to(config.PROJECT_ROOT)}' to persistent cache...")
            for item_name in os.listdir(project_benchmark_dir):
                shutil.move(str(project_benchmark_dir / item_name), str(cache_benchmark_dir / item_name))
            project_benchmark_dir.rmdir()
            self._create_link_or_copy_dir(cache_benchmark_dir, project_benchmark_dir)
            print("  Migration complete.")
        elif not project_benchmark_dir.exists(): # --- FIX: Make log message more general to cover the copy fallback case ---
            self._create_link_or_copy_dir(cache_benchmark_dir, project_benchmark_dir)
            print(f"  Project benchmark directory linked to persistent cache.")
        return str(project_benchmark_dir)

    def _get_dataset(self, name: str, **kwargs) -> Optional[Any]:
        """Loads a standard PyG dataset."""
        try:
            if name in ['Cora', 'CiteSeer', 'PubMed']: return Planetoid(root=self.dataset_root, name=name, **kwargs)
            if name in ['Cornell', 'Texas', 'Wisconsin']: return WebKB(root=self.dataset_root, name=name, **kwargs)
            if name == 'Actor': return Actor(root=self.dataset_root, **kwargs) # noqa
            if name == 'KarateClub': return KarateClub(**kwargs)
            raise ValueError(f"Dataset '{name}' not recognized.")
        except Exception as e:
            print(f"  Error loading dataset '{name}': {e.__class__.__name__}: {e}")
            return None

    def _get_1d_mask(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """Helper to handle masks from datasets that may have multiple splits (e.g., WebKB)."""
        return mask_tensor[:, 0].bool() if mask_tensor.dim() > 1 else mask_tensor.bool()

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Main execution function for the benchmarker."""
        raise NotImplementedError

    @staticmethod
    def display_aggregated_benchmark_summary(all_results: List[pd.DataFrame]):
        """
        Standardizes, concatenates, and displays a final summary of all benchmark results.
        DirectGCN rows are disambiguated by appending the short variant tag (if present).
        """
        if not all_results:
            print("No benchmark results were generated to aggregate.")
            return

        # Make the console table readable
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_colwidth', None)

        try:
            final_summary_df = pd.concat(all_results, ignore_index=True)

            # Compute dataset grouping (collapse _Original/_Undirected into one group)
            def get_group_name(dataset_name):
                if not isinstance(dataset_name, str):
                    return "Unknown"
                if 'ProtGram_n1_Singleton' in dataset_name:
                    return dataset_name
                return dataset_name.replace('_Original', '')

            final_summary_df['dataset_group'] = final_summary_df['dataset'].apply(get_group_name)

            # Ensure model column is string for safe suffix appending
            final_summary_df['model'] = final_summary_df['model'].astype(str)

            # Append variant suffix to DirectGCN display name when available
            if 'variant' in final_summary_df.columns:
                mask_dgcn = final_summary_df['model'] == 'DirectGCN'
                has_variant = mask_dgcn & final_summary_df['variant'].notna() & (final_summary_df['variant'].astype(str).str.strip() != '')
                final_summary_df.loc[has_variant, 'model'] = (
                    final_summary_df.loc[has_variant, 'model'] + '[' + final_summary_df.loc[has_variant, 'variant'].astype(str) + ']'
                )

            # Columns to print
            columns_to_print = [
                'model', 'Accuracy', 'F1-Score (Macro)',
                'Precision (Macro)', 'Recall (Macro)', 'error'
            ]
            # Backfill missing columns for robustness
            for col in columns_to_print:
                if col not in final_summary_df.columns:
                    final_summary_df[col] = 'N/A'

            # Sort within each dataset group by model name for stable display
            final_summary_df = final_summary_df.sort_values(by=['dataset_group', 'model'])

            DataUtils.print_header("Aggregated Benchmark Summary")
            for group_name, group_df in final_summary_df.groupby('dataset_group', sort=False):
                print(f"\n--- Results for Dataset: {group_name} ---")
                formatted_group_df = group_df.copy()

                # Format floats to four decimals; leave non-floats as-is
                float_cols = formatted_group_df.select_dtypes(include=['float']).columns
                for col in float_cols:
                    formatted_group_df[col] = formatted_group_df[col].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')

                print(formatted_group_df[columns_to_print].to_string(index=False, na_rep='N/A'))
        except Exception as e:
            import traceback
            print("\n--- ❌ ERROR: Could not generate the aggregated benchmark summary. ---")
            print("This can happen if the results dataframes have an unexpected structure or contain invalid data.")
            print(f"Error details: {e}")
            traceback.print_exc()