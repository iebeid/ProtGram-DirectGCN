# ==============================================================================
# MODULE: configuration/processor.py
# PURPOSE: Handles the verification and acquisition of all external data files.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import hashlib
from pathlib import Path
from tqdm.auto import tqdm
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# --- NEW: Import PyG for benchmark dataset downloading ---

# Conditionally import gdown to avoid making it a hard dependency
try:
    import gdown

    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False


class DataProcessor:
    """
    Handles the download, processing, and validation of all project data.
    This class orchestrates the creation of a clean, analysis-ready set of
    Parquet files from various raw data sources.
    """

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _is_file_valid(file_path: Path) -> bool:
        """
        Performs a basic integrity check on a file beyond just existence.
        """
        return file_path.exists() and file_path.stat().st_size > 100

    @staticmethod
    def _calculate_sha256(file_path: Path) -> str:
        """Calculates the SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except (IOError, OSError):
            return ""

    def _process_uniprot_mapping(self):
        """Processes the raw UniProt ID mapping file into a filtered Parquet file."""
        # --- NEW: Add caching for the processed parquet file to avoid re-processing ---
        processed_parquet_path = self.config.ID_MAPPING_PATH
        cache_path = self.config.PERSISTENT_DATA_CACHE / processed_parquet_path.name

        # 1. Check if a valid cached version exists and restore it.
        if cache_path.exists():
            print(f"☑ Found cached processed ID mapping file: {cache_path.name}")
            processed_parquet_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cache_path, processed_parquet_path)
            print(f"  Restored {processed_parquet_path.name} from cache.")
            return

        # 2. Check if a local version already exists (and cache it for next time).
        if processed_parquet_path.exists():
            print(f"☑ Found existing processed ID mapping file: {processed_parquet_path.name}")
            print("  Copying to cache for future runs...")
            shutil.copy(processed_parquet_path, cache_path)
            return

        # 3. If neither exists, run the full, memory-intensive processing.
        print("\n--- Step 2a: Processing UniProt ID Mapping File ---")
        raw_mapping_path = self.config.DATA_SOURCES['UNIPROT_ID_MAPPING']['path']
        if not DataProcessor._is_file_valid(raw_mapping_path):
            print(f"  ERROR: Raw UniProt mapping file not found at {raw_mapping_path}. Cannot proceed.")
            return

        # --- DEFINITIVE FIX for OOM Kill: Use a two-stage, memory-efficient streaming approach --- #
        # Stage 1: Stream the huge raw file line-by-line, writing filtered lines to a temporary file. This uses minimal RAM.
        print(f"  Stage 1/2: Streaming and filtering raw mapping file '{raw_mapping_path.name}'...")
        temp_filtered_path = self.config.ID_MAPPING_PATH.with_suffix('.tmp.tsv')
        relevant_dbs = ['GeneID', 'UniRef100', 'UniRef90', 'UniRef50']
        total_size = raw_mapping_path.stat().st_size
        lines_written = 0
        with open(raw_mapping_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
             open(temp_filtered_path, 'w', encoding='utf-8') as f_out, \
             tqdm(total=total_size, unit='B', unit_scale=True, desc="  - Filtering raw data") as pbar:
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[1] in relevant_dbs:
                    f_out.write(line)
                    lines_written += 1
                pbar.update(len(line.encode('utf-8')))

        if lines_written == 0:
            print("  - WARNING: No relevant IDs found in the mapping file. The resulting Parquet file will be empty.")
            empty_df = pd.DataFrame(columns=['uniprot_id', 'db', 'other_id'])
            empty_df.to_parquet(self.config.ID_MAPPING_PATH, engine='pyarrow')
            temp_filtered_path.unlink() # Clean up temp file
            return

        # Stage 2: Convert the much smaller temporary file to a partitioned Parquet file.
        print(f"\n  Stage 2/2: Converting {lines_written:,} filtered lines to Parquet format...")
        ddf = dd.read_csv(temp_filtered_path, sep='\t', header=None, names=['uniprot_id', 'db', 'other_id'], blocksize='256MB')
        ddf.to_parquet(self.config.ID_MAPPING_PATH, engine='pyarrow', overwrite=True)
        temp_filtered_path.unlink() # Clean up the intermediate file

        # --- NEW: Cache the newly created file for future runs ---
        if processed_parquet_path.exists():
            print(f"  Caching new {processed_parquet_path.name} for future runs...")
            shutil.copy(processed_parquet_path, cache_path)

        print("  ✔ UniProt ID mapping processing complete.")

    def _process_negative_interactions(self):
        """Combines and processes all raw negative interaction files into a single Parquet file."""
        if self.config.NEG_INTERACTIONS_PATH.exists():
            print(f"☑ Found existing processed negative interactions file: {self.config.NEG_INTERACTIONS_PATH.name}")
            return

        print("\n--- Step 2b: Processing Negative Interaction Files ---")
        neg_files = [v['path'] for k, v in self.config.DATA_SOURCES.items() if k.startswith('NEG_INTERACTIONS')]
        existing_neg_files = [f for f in neg_files if DataProcessor._is_file_valid(f)]

        if not existing_neg_files:
            print("  ERROR: No raw negative interaction files found. Cannot proceed.")
            return

        with ProgressBar():
            print(f"  Reading {len(existing_neg_files)} negative interaction files...")
            ddf = dd.read_csv(existing_neg_files, sep='\t', header=None, usecols=[0, 1],
                              on_bad_lines='warn', blocksize='64MB', dtype=str)
            ddf.columns = ['p1_raw', 'p2_raw']

            # Extract UniProtKB ID from 'uniprotkb:ID' format
            ddf['p1'] = ddf['p1_raw'].str.split(':').str[1]
            ddf['p2'] = ddf['p2_raw'].str.split(':').str[1]

            final_ddf = ddf[['p1', 'p2']].dropna().repartition(npartitions=1)
            print(f"  Saving processed negative interactions to {self.config.NEG_INTERACTIONS_PATH.name}...")
            final_ddf.to_parquet(self.config.NEG_INTERACTIONS_PATH, engine='pyarrow', overwrite=True)
        print("  ✔ Negative interaction processing complete.")

    def _process_biogrid_interactions(self):
        """Processes BioGRID interactions, mapping GeneIDs to UniProtKB IDs."""
        if self.config.POS_INTERACTIONS_PATH.exists():
            print(f"☑ Found existing processed positive interactions file: {self.config.POS_INTERACTIONS_PATH.name}")
            return

        print("\n--- Step 2c: Processing BioGRID Positive Interactions ---")
        raw_biogrid_path = self.config.DATA_SOURCES['BIOGRID_INTERACTIONS']['path']
        id_mapping_path = self.config.ID_MAPPING_PATH

        if not DataProcessor._is_file_valid(raw_biogrid_path) or not id_mapping_path.exists():
            print("  ERROR: Raw BioGRID file or processed ID mapping file not found. Cannot proceed.")
            return

        with ProgressBar():
            print(f"  Reading BioGRID data from {raw_biogrid_path.name}...")
            biogrid_ddf = dd.read_csv(raw_biogrid_path, sep='\t', header=0, usecols=[0, 1],
                                      on_bad_lines='warn', blocksize='128MB', dtype=str)
            biogrid_ddf.columns = ['p1_raw', 'p2_raw']

            # Extract GeneID from 'entrez gene/locuslink:ID' format
            biogrid_ddf['gene_id_1'] = biogrid_ddf['p1_raw'].str.split(':').str[1]
            biogrid_ddf['gene_id_2'] = biogrid_ddf['p2_raw'].str.split(':').str[1]
            biogrid_pairs_ddf = biogrid_ddf[['gene_id_1', 'gene_id_2']].dropna().astype(
                {'gene_id_1': 'int64', 'gene_id_2': 'int64'})

            print(f"  Reading ID mapping from {id_mapping_path.name}...")
            mapping_ddf = dd.read_parquet(id_mapping_path)
            geneid_to_uniprot_map = mapping_ddf[mapping_ddf['db'] == 'GeneID'].drop('db', axis=1)
            geneid_to_uniprot_map['other_id'] = geneid_to_uniprot_map['other_id'].astype('int64')

            # Persist the smaller mapping table for faster joins
            geneid_to_uniprot_map = geneid_to_uniprot_map.persist()

            print("  Performing two-way merge to map GeneIDs to UniProtKB IDs...")
            # Merge for the first protein
            merged1 = dd.merge(biogrid_pairs_ddf, geneid_to_uniprot_map, left_on='gene_id_1', right_on='other_id',
                               how='inner')
            merged1 = merged1.rename(columns={'uniprot_id': 'p1'}).drop(['gene_id_1', 'other_id'], axis=1)

            # Merge for the second protein
            merged2 = dd.merge(merged1, geneid_to_uniprot_map, left_on='gene_id_2', right_on='other_id', how='inner')
            merged2 = merged2.rename(columns={'uniprot_id': 'p2'}).drop(['gene_id_2', 'other_id'], axis=1)

            final_ddf = merged2[['p1', 'p2']].drop_duplicates().repartition(npartitions=4)

            print(f"  Saving final mapped positive interactions to {self.config.POS_INTERACTIONS_PATH.name}...")
            final_ddf.to_parquet(self.config.POS_INTERACTIONS_PATH, engine='pyarrow', overwrite=True)
        print("  ✔ Positive interaction processing complete.")
