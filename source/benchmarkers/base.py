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
from source.utils.data import DataUtils


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
            os.symlink(source, dest, target_is_directory=True)
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
        elif not project_benchmark_dir.exists():
            self._create_link_or_copy_dir(cache_benchmark_dir, project_benchmark_dir)
            print(f"  Symlinked project benchmark directory to persistent cache.")
        return str(project_benchmark_dir)

    def _get_dataset(self, name: str, **kwargs):
        """Loads a standard PyG dataset."""
        try:
            if name in ['Cora', 'CiteSeer', 'PubMed']: return Planetoid(root=self.dataset_root, name=name, **kwargs)
            if name in ['Cornell', 'Texas', 'Wisconsin']: return WebKB(root=self.dataset_root, name=name, **kwargs)
            if name == 'Actor': return Actor(root=self.dataset_root, **kwargs)
            if name == 'KarateClub': return KarateClub(root=self.dataset_root, **kwargs)
            raise ValueError(f"Dataset '{name}' not recognized.")
        except Exception as e:
            print(f"  Error loading dataset '{name}': {e}")
            return None

    def _get_1d_mask(self, mask_tensor: torch.Tensor) -> torch.Tensor:
        """Helper to handle masks from datasets that may have multiple splits (e.g., WebKB)."""
        return mask_tensor[:, 0].bool() if mask_tensor.dim() > 1 else mask_tensor.bool()

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Main execution function for the benchmarker."""
        raise NotImplementedError