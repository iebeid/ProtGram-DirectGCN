# ==============================================================================
# MODULE: experiments/helpers.py
# PURPOSE: Contains helper functions for experiments, such as dummy data generation.
# VERSION: 1.0 (Initial creation)
# AUTHOR: Gemini Code Assist
# ==============================================================================

import os
import random
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple

import h5py
import numpy as np
import pandas as pd


def create_dummy_data_for_ppi_eval(
        base_dir: Path,
        num_proteins: int,
        embedding_dim: int,
        num_pos: int,
        num_neg: int
) -> Tuple[Path, Path, List[Dict[str, Any]]]:
    """
    Creates a complete set of dummy data for a PPI evaluation run.

    This includes dummy protein embeddings in an H5 file and dummy positive/negative
    interaction pairs in CSV files.

    Args:
        base_dir (Path): The base directory to create the 'dummy_data_temp' folder in.
        num_proteins (int): The number of dummy protein IDs to generate.
        embedding_dim (int): The dimensionality of the dummy embeddings.
        num_pos (int): The number of positive interaction pairs to generate.
        num_neg (int): The number of negative interaction pairs to generate.

    Returns:
        A tuple containing:
        - The file path to the positive interactions CSV.
        - The file path to the negative interactions CSV.
        - A list containing the configuration dictionary for the dummy embedding file.
    """
    dummy_data_dir = base_dir / "dummy_data_temp"
    if dummy_data_dir.exists():
        shutil.rmtree(dummy_data_dir)
    dummy_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Creating dummy data in: {dummy_data_dir} (Proteins: {num_proteins}, Dim: {embedding_dim}, Pos: {num_pos}, Neg: {num_neg})")

    protein_ids = [f"DUMMY_P{i:04d}" for i in range(num_proteins)]

    # Create dummy embeddings
    dummy_emb_file = dummy_data_dir / "dummy_embeddings.h5"
    with h5py.File(dummy_emb_file, 'w') as hf:
        for pid in protein_ids:
            hf.create_dataset(pid, data=np.random.rand(embedding_dim).astype(np.float16))
    print(f"  Dummy embeddings saved to: {dummy_emb_file}")

    # Create dummy positive interactions
    dummy_pos_path = dummy_data_dir / "dummy_pos.csv"
    pos_pairs = pd.DataFrame([random.sample(protein_ids, 2) for _ in range(num_pos)], columns=['p1', 'p2'])
    pos_pairs.to_csv(dummy_pos_path, header=False, index=False)
    print(f"  Dummy positive interactions saved to: {dummy_pos_path}")

    # Create dummy negative interactions
    dummy_neg_path = dummy_data_dir / "dummy_neg.csv"
    neg_pairs = pd.DataFrame([random.sample(protein_ids, 2) for _ in range(num_neg)], columns=['p1', 'p2'])
    neg_pairs.to_csv(dummy_neg_path, header=False, index=False)
    print(f"  Dummy negative interactions saved to: {dummy_neg_path}")

    dummy_emb_config = [{"path": str(dummy_emb_file), "name": "DummyEmb"}]
    return dummy_pos_path, dummy_neg_path, dummy_emb_config