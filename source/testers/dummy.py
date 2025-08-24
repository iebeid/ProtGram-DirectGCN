# ==============================================================================
# MODULE: testers/dummy.py
# PURPOSE: Contains helper functions for creating dummy data for testing.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
from typing import Optional, List
import numpy as np
import pandas as pd
import h5py

class DummyDataFactory:
    """A factory class for creating dummy data for testing purposes."""

    @staticmethod
    def create_fasta(directory: str, filename: str = "dummy_test.fasta", num_seqs: int = 5):
        """Creates a small dummy FASTA file for testing purposes."""
        os.makedirs(directory, exist_ok=True)
        fasta_path = os.path.join(directory, filename)
        with open(fasta_path, "w") as f:
            for i in range(num_seqs):
                # --- REFACTOR: Use a more realistic UniProt-like header ---
                seq_id = f"sp|DUMMY{i+1:04d}|TEST_DUMMY"
                sequence = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=np.random.randint(20, 50)))
                f.write(f">{seq_id}\n{sequence}\n")
        return fasta_path

    @staticmethod
    def create_interaction_files(directory: str, num_pairs: int = 10, num_proteins: int = 20):
        """Creates dummy positive and negative interaction files."""
        os.makedirs(directory, exist_ok=True)
        protein_ids = [f"P{i:03d}" for i in range(num_proteins)]
        pos_path = os.path.join(directory, "dummy_positive_interactions.csv")
        neg_path = os.path.join(directory, "dummy_negative_interactions.csv")

        def generate_pairs(filepath, count):
            pairs = set()
            max_possible = num_proteins * (num_proteins - 1) // 2
            actual_count = min(count, max_possible)
            attempts = 0
            while len(pairs) < actual_count and attempts < count * 5:
                if num_proteins < 2: break
                p1, p2 = np.random.choice(protein_ids, 2, replace=False)
                pairs.add(tuple(sorted((p1, p2))))
                attempts += 1
            pd.DataFrame(list(pairs)).to_csv(filepath, header=False, index=False)

        generate_pairs(pos_path, num_pairs)
        generate_pairs(neg_path, num_pairs)
        return pos_path, neg_path

    @staticmethod
    def create_dummy_id_mapping_parquet(directory: str, filename: str = "dummy_id_mapping.parquet", num_ids: int = 5):
        """Creates a small dummy ID mapping Parquet file for testing."""
        os.makedirs(directory, exist_ok=True)
        parquet_path = os.path.join(directory, filename) # --- DEFINITIVE FIX: Create dummy data that matches the real processed format ---
        # The IDMapGenerator expects 'other_id' and 'uniprot_id' before it renames them.
        ids = [f"DUMMY{i+1:04d}" for i in range(num_ids)]
        df = pd.DataFrame({
            'other_id': ids,
            'uniprot_id': ids  # Simple 1-to-1 mapping is sufficient for the test
        })
        df.to_parquet(parquet_path)
        return parquet_path

    @staticmethod
    def create_h5_embeddings(directory: str, filename: str = "dummy_embeddings.h5", protein_ids: Optional[List[str]] = None, num_proteins: int = 20, dim: int = 10):
        """Creates a dummy H5 embedding file."""
        os.makedirs(directory, exist_ok=True)
        h5_path = os.path.join(directory, filename)
        if protein_ids is None:
            protein_ids = [f"DUMMY_P{i:05d}" for i in range(num_proteins)]
        with h5py.File(h5_path, 'w') as hf:
            for pid in protein_ids:
                # --- FIX: Use float16 to match the data type used in the main pipeline's dummy run ---
                hf.create_dataset(pid, data=np.random.rand(dim).astype(np.float16))
        return h5_path