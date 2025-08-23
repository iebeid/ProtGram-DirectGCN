# ==============================================================================
# MODULE: utils/data/data_utils.py
# PURPOSE: Contains general, non-I/O data loading and processing utilities.
# VERSION: 4.1 (Added memory reporting utility)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import random
import shutil
from pathlib import Path
from typing import Iterator, List, Optional, Any, Tuple

import h5py
import numpy as np
import pandas as pd
import psutil
import torch


# ==============================================================================
# 1. General Data Utilities (Moved from embedding_loader.py)
# ==============================================================================
class DataUtils:
    """General data utility functions."""

    @staticmethod
    def set_seeds(seed: int):
        """
        Sets random seeds for all relevant libraries to ensure reproducibility.
        Also configures PyTorch to use deterministic algorithms for CUDA.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # --- DEFINITIVE FIX for Reproducibility ---
        # This forces PyTorch to use deterministic algorithms, which is essential for run-to-run consistency.
        # It may impact performance slightly but is crucial for reliable experiments.
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # The following two lines are crucial for GPU reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"  Seeds set to {seed} for reproducibility. PyTorch CUDA deterministic mode is ON.")

    @staticmethod
    def reservoir_sample(iterator: Iterator[Any], k: int, random_seed: Optional[int] = None) -> List[Any]:
        """
        Performs reservoir sampling on a potentially very large iterator.
        This allows taking a random sample without loading the entire iterator into memory.
        """
        if random_seed is not None:
            random.seed(random_seed)

        reservoir = []
        for i, item in enumerate(iterator):
            if i < k:
                reservoir.append(item)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = item
        return reservoir

    @staticmethod
    def print_header(title: str):
        """Prints a standardized header to the console."""
        border = "=" * (len(title) + 6)
        print(f"\n{border}\n### {title} ###\n{border}\n")

    @staticmethod
    def report_system_resources(output_path: Path):
        """Prints a detailed report of available system resources as a pre-flight check."""
        DataUtils.print_header("System Resource Pre-flight Check")

        # RAM
        ram = psutil.virtual_memory()
        print(f"RAM: {ram.available / (1024**3):.2f} GB Available / {ram.total / (1024**3):.2f} GB Total")

        # Disk Space
        try:
            # Ensure the directory exists to check its disk
            output_path.mkdir(parents=True, exist_ok=True)
            disk = shutil.disk_usage(output_path)
            print(f"Disk ({output_path}): {disk.free / (1024**3):.2f} GB Free / {disk.total / (1024**3):.2f} GB Total")
        except Exception as e:
            print(f"Disk ({output_path}): Could not check disk space. Error: {e}")

        # GPU
        if torch.cuda.is_available():
            print("GPU(s):")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_mem_gb = props.total_memory / (1024**3)
                print(f"  - GPU {i} ({props.name}): {total_mem_gb:.2f} GB Total Memory")
        else:
            print("GPU: Not available. Running on CPU.")
        print("-" * (len("System Resource Pre-flight Check") + 6) + "\n")

    @staticmethod
    def report_memory_usage(context: str):
        """Prints current CPU and GPU memory usage for debugging."""
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss / (1024 * 1024)  # in MB
        print(f"--- Memory Usage ({context}): RSS = {rss:.2f} MB ---")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            print(f"  GPU Memory: Allocated = {allocated:.2f} MB, Reserved = {reserved:.2f} MB")

    @staticmethod
    def has_enough_memory(required_gb: float, context: str) -> bool:
        """
        Checks if the system has enough available memory for an operation.
        This is a proactive check to avoid an OS-level OOM kill.

        Returns:
            True if memory is sufficient, False otherwise.
        """
        available_mem_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_mem_gb < required_gb:
            print("\n" + "!" * 80)
            print(f"!!! MEMORY WARNING ({context}) !!!")
            print(f"  Operation requires an estimated {required_gb:.2f} GB, but only {available_mem_gb:.2f} GB is available.")
            print("  The pipeline will attempt to skip this step gracefully to prevent a system-wide OOM error.")
            print("!" * 80 + "\n")
            return False
        return True

    @staticmethod
    def create_dummy_embedding_file(output_dir: str, filename: str, protein_ids: List[str], embedding_dim: int) -> str:
        """Creates a dummy HDF5 embedding file for testing the main pipeline."""
        h5_path = os.path.join(output_dir, filename)
        with h5py.File(h5_path, 'w') as hf:
            for pid in protein_ids:
                hf.create_dataset(pid, data=np.random.rand(embedding_dim).astype(np.float16))
        print(f"  Dummy embeddings saved to: {h5_path}")
        return h5_path

    @staticmethod
    def create_dummy_interaction_files(output_dir: str, protein_ids: List[str], num_pos: int, num_neg: int) -> Tuple[str, str]:
        """Creates dummy positive and negative interaction CSV files."""
        pos_path = os.path.join(output_dir, "dummy_pos.csv")
        neg_path = os.path.join(output_dir, "dummy_neg.csv")

        # Ensure we don't try to sample more pairs than possible
        if len(protein_ids) < 2:
            pos_pairs = pd.DataFrame(columns=['p1', 'p2'])
            neg_pairs = pd.DataFrame(columns=['p1', 'p2'])
        else:
            # Ensure we don't create duplicate pairs
            pos_pairs_set = {tuple(sorted(pair)) for pair in [random.sample(protein_ids, 2) for _ in range(num_pos * 2)]}
            neg_pairs_set = {tuple(sorted(pair)) for pair in [random.sample(protein_ids, 2) for _ in range(num_neg * 2)]}
            pos_pairs = pd.DataFrame(list(pos_pairs_set)[:num_pos], columns=['p1', 'p2'])
            neg_pairs = pd.DataFrame(list(neg_pairs_set - pos_pairs_set)[:num_neg], columns=['p1', 'p2'])

        pos_pairs.to_csv(pos_path, header=False, index=False)
        print(f"  Dummy positive interactions saved to: {pos_path}")
        neg_pairs.to_csv(neg_path, header=False, index=False)
        print(f"  Dummy negative interactions saved to: {neg_path}")
        return pos_path, neg_path
