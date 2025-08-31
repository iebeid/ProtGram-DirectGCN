# ==============================================================================
# MODULE: configuration/processor.py
# PURPOSE: Processes raw data files into a unified Parquet format.
# VERSION: 9.0 (Streamlined by removing obsolete ID Mapping logic)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import hashlib
from pathlib import Path
import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar

class DataProcessor:
    """
    Handles the conversion of raw interaction data files into standardized
    and efficient Parquet files for downstream analysis.
    """
    def __init__(self, config):
        self.config = config

    def _process_negative_interactions(self):
        """Processes and combines all negative interaction files."""
        

        print("  - Processing negative interaction files...")
        dfs = []
        for file_path in self.config.NEG_INTERACTIONS_RAW_PATHS:
            if file_path.exists():
                try:
                    ddf = dd.read_csv(
                        file_path, sep='\t', usecols=[0, 1], header=None,
                        names=['protein1', 'protein2'], dtype='object', on_bad_lines='warn'
                    )
                    
                    def clean_uniprot_id(df):
                        df['protein1'] = df['protein1'].str.replace('uniprot:', '', regex=False)
                        df['protein2'] = df['protein2'].str.replace('uniprot:', '', regex=False)
                        return df
                    
                    meta = pd.DataFrame(columns=['protein1', 'protein2'], dtype=str)
                    ddf = ddf.map_partitions(clean_uniprot_id, meta=meta)
                    dfs.append(ddf)
                except Exception as e:
                    print(f"    - WARNING: Could not process file {file_path.name}: {e}")

        if not dfs:
            raise FileNotFoundError("No valid negative interaction files were found or processed.")

        combined_ddf = dd.concat(dfs, axis=0).drop_duplicates().repartition(npartitions=self.config.DASK_N_PARTITIONS)
        print(f"  - Saving combined negative interactions to: {self.config.NEG_INTERACTIONS_PATH.name}")
        with ProgressBar():
            combined_ddf.to_parquet(self.config.NEG_INTERACTIONS_PATH, engine='pyarrow', overwrite=True)

    def _process_biogrid_interactions(self):
        """Processes the BioGRID MITAB file to extract positive protein interactions."""
        

        source_path = self.config.BIOGRID_RAW_PATH
        if not source_path.exists():
            raise FileNotFoundError(f"BioGRID source file not found at {source_path}")

        print("  - Processing BioGRID MITAB file...")
        ddf = dd.read_csv(
            source_path, sep='\t', header=0,
            usecols=['#ID Interactor A', 'ID Interactor B', 'Taxid Interactor A', 'Taxid Interactor B'],
            dtype='object', on_bad_lines='warn'
        )
        human_interactions = ddf[(ddf['Taxid Interactor A'] == 'taxid:9606') & (ddf['Taxid Interactor B'] == 'taxid:9606')]
        
        def extract_uniprot_id(df):
            df['protein1'] = df['#ID Interactor A'].str.split(':').str[1]
            df['protein2'] = df['ID Interactor B'].str.split(':').str[1]
            return df[['protein1', 'protein2']]

        meta = pd.DataFrame(columns=['protein1', 'protein2'], dtype=str)
        final_ddf = human_interactions.map_partitions(extract_uniprot_id, meta=meta)
        
        final_ddf = final_ddf.dropna().drop_duplicates().repartition(npartitions=self.config.DASK_N_PARTITIONS)

        print(f"  - Saving positive interactions to: {self.config.POS_INTERACTIONS_PATH.name}")
        with ProgressBar():
            final_ddf.to_parquet(self.config.POS_INTERACTIONS_PATH, engine='pyarrow', overwrite=True)

    @staticmethod
    def _is_file_valid(file_path: Path) -> bool:
        """Checks if a file exists and is not empty."""
        return file_path.exists() and file_path.stat().st_size > 0

    @staticmethod
    def _calculate_sha256(file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def is_huge_file(self, file_path: Path) -> bool:
        # --- DEFINITIVE FIX: Use the correct configuration attribute and compare bytes to bytes ---
        # The previous logic was comparing bytes to gigabytes and using a non-existent config variable.
        return file_path.stat().st_size > self.config.CHECKSUM_SKIP_SIZE_BYTES
