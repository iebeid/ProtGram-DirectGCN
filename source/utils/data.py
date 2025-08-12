# ==============================================================================
# MODULE: utils/data.py
# PURPOSE: Contains all data loading and processing utilities.
# VERSION: 4.0 (Added global seeding and reservoir sampling)
# AUTHOR: Islam Ebeid
# ==============================================================================

import os
import json
import pickle
import random
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union, Any

import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import requests
from Bio import SeqIO
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

from configuration.config import Config
from source.utils.models import EmbeddingProcessor


# ==============================================================================
# 1. General Data Utilities (Moved from post.py)
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
    def print_header(title: str):
        border = "=" * (len(title) + 6)
        print(f"\n{border}\n### {title} ###\n{border}\n")

    @staticmethod
    def save_object(obj: any, filepath: Union[str, Path]):
        filepath = Path(filepath)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
            print(f"Object saved to {filepath}")
        except Exception as e:
            print(f"Error saving object to {filepath}: {e}")

    @staticmethod
    def load_object(filepath: Union[str, Path]) -> Optional[any]:
        filepath = Path(filepath)
        if not filepath.exists(): return None
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading object from {filepath}: {e}")
            return None

    @staticmethod
    def save_dataframe_to_csv(df: pd.DataFrame, output_path: Union[str, Path], index: bool = False):
        output_path = Path(output_path)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=index)
            print(f"DataFrame saved to: {output_path}")
        except Exception as e:
            print(f"Error saving DataFrame to {output_path}: {e}")

    @staticmethod
    def save_json(data: Dict, filepath: Union[str, Path]):
        """Saves a dictionary to a JSON file with pretty printing."""
        filepath = Path(filepath)
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, sort_keys=True)
            print(f"  JSON data saved to {filepath.name}")
        except Exception as e:
            print(f"  ERROR: Could not save JSON to {filepath.name}: {e}")

    @staticmethod
    def write_h5(embeddings_dict: Dict, path: Path, desc: str):
        """Helper function to write a dictionary of embeddings to an HDF5 file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, 'w') as hf:
            for key, value in tqdm(embeddings_dict.items(), desc=f"  {desc}"):
                if value is not None:
                    hf.create_dataset(key, data=value)

    @staticmethod
    def check_h5_embeddings_integrity(h5_filepath: Union[str, Path], num_samples_to_check: int = 5):
        h5_filepath = Path(h5_filepath)
        DataUtils.print_header(f"Checking HDF5 file: {h5_filepath.name}")
        if not h5_filepath.exists() or not h5py.is_hdf5(h5_filepath):
            print(f"Error: File at '{h5_filepath}' is not a valid HDF5 file or does not exist.")
            return
        try:
            with h5py.File(h5_filepath, 'r') as hf:
                keys = list(hf.keys())
                if not keys:
                    print("HDF5 check: File is empty.")
                    return
                print(f"Found {len(keys)} total embeddings. Inspecting up to {num_samples_to_check} samples:")
                sample_keys = random.sample(keys, min(len(keys), num_samples_to_check))
                for i, key in enumerate(sample_keys):
                    dataset = hf.get(key)
                    if not isinstance(dataset, h5py.Dataset): continue
                    emb = dataset[:]
                    print(f"  - Sample {i + 1}: Key='{key}', Shape={emb.shape}, DType={emb.dtype}")
                    if np.isnan(emb).any(): print("    - WARNING: Embedding contains NaN values.")
                    if np.isinf(emb).any(): print("    - WARNING: Embedding contains Inf values.")
        except Exception as e:
            print(f"An error occurred while checking HDF5 file '{h5_filepath}': {e}")

    @staticmethod
    def get_id_mapping(config: Config) -> Dict[str, str]:
        """
        Loads the UniProt ID mapping file into a dictionary.
        """
        id_map = {}
        mapping_file = config.ID_MAPPING_PATH
        if not mapping_file.exists():
            print(f"  - WARNING: ID mapping file not found at {mapping_file}. Returning empty map.")
            return id_map

        print(f"  Loading Protein ID Mapping from: {mapping_file.name}")
        with open(mapping_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    from_id, to_id = parts
                    id_map[from_id] = to_id
        print(f"  ID mapping loaded with {len(id_map)} entries.")
        return id_map

    def save_final_embeddings(self, embeddings_per_model: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, str]:
        """Saves final protein embeddings and their PCA versions to H5 files."""
        output_paths = {}
        for model_type, embeddings in embeddings_per_model.items():
            if not embeddings:
                print(f"  No embeddings generated for model '{model_type}'. Skipping save.")
                continue

            model_name = f"ProtGram{model_type.capitalize()}"
            output_dir = self.config.RESULTS_GCN_EMBEDDINGS_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{model_name}.h5"

            DataUtils.write_h5(embeddings, str(output_path), f"Writing H5 for {model_name}")
            output_paths[model_name] = str(output_path)

            # Apply PCA and save
            pca_path = EmbeddingProcessor.apply_pca_to_h5(
                input_h5_path=output_path,
                output_dir=output_dir,
                target_dimension=self.config.PCA_TARGET_DIMENSION,
                random_seed=self.config.RANDOM_STATE
            )
            if str(pca_path) != str(output_path):
                output_paths[f"{model_name}_pca"] = str(pca_path)

        return output_paths

    @staticmethod
    def save_embeddings(model: torch.nn.Module, data: Data, config: Config, embedding_dir: Path, device: torch.device):
        """Extracts, processes (with PCA), and saves embeddings."""
        print(f"    Extracting embeddings for {model.__class__.__name__}...")
        with torch.no_grad():
            model.eval()
            _, embeddings = model(data.to(device))

        if embeddings is None:
            print("    Warning: Could not extract embeddings.")
            return

        embeddings_np = embeddings.cpu().numpy()
        final_embedding_dim = embeddings_np.shape[1]
        output_suffix = f"_dim{final_embedding_dim}"

        if config.BENCHMARK_APPLY_PCA_TO_EMBEDDINGS and embeddings_np.shape[0] > config.BENCHMARK_PCA_TARGET_DIM:
            print(f"      Applying PCA (target dim: {config.BENCHMARK_PCA_TARGET_DIM})...")
            embeddings_for_pca = {i: emb for i, emb in enumerate(embeddings_np)}
            pca_embed_dict = EmbeddingProcessor.apply_pca(embeddings_for_pca, config.BENCHMARK_PCA_TARGET_DIM, config.RANDOM_STATE)
            if pca_embed_dict:
                embeddings_np = np.array(list(pca_embed_dict.values()))
                final_embedding_dim = embeddings_np.shape[1]
                output_suffix = f"_pca{final_embedding_dim}"

        emb_dict = {str(i): embeddings_np[i] for i in range(embeddings_np.shape[0])}
        save_path_emb_dir = embedding_dir / data.name
        save_path_emb_dir.mkdir(parents=True, exist_ok=True)
        h5_path = save_path_emb_dir / f"{model.__class__.__name__}_embeddings{output_suffix}.h5"
        DataUtils.write_h5(emb_dict, h5_path, f"Writing H5 for {model.__class__.__name__}")
        print(f"      Saved embeddings to {h5_path}")


# ==============================================================================
# 2. FASTA File Utilities
# ==============================================================================
class FastaUtils:
    """
    A collection of utilities for handling FASTA files, including an
    efficient parser and a memory-safe corpus class for sequence processing.
    """
    AMINO_ACID_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

    @staticmethod
    def extract_id_from_header(header: str) -> str:
        """
        Robustly extracts a protein identifier from a FASTA header.
        Prioritizes UniProt accession numbers but falls back to the first word.
        """
        hid = header.strip().lstrip('>')
        # Regex for standard UniProt headers (e.g., >sp|P12345|ID_NAME)
        up_match = re.match(r"^(?:sp|tr)\|([OPQ]?[A-Z0-9]{5,9}(?:-\d+)?)\|", hid, re.IGNORECASE)
        if up_match:
            return up_match.group(1)
        # Regex for UniRef headers (e.g., >UniRef90_A0A0A0A0A0)
        uniref_match = re.match(r"^(?:UniRef\d{2,3})_([A-Z0-9]+)", hid, re.IGNORECASE)
        if uniref_match:
            return uniref_match.group(1)  # Return the accession, not the UniRef ID itself
        # Fallback to the first word in the header
        return hid.split()[0]

    @staticmethod
    def parse_sequences(fasta_filepaths: List[Union[str, Path]]) -> Iterator[Tuple[str, str]]:
        """
        An efficient FASTA parser that reads one or more FASTA files, yielding
        an ID and sequence for each record.
        """
        for path_str in fasta_filepaths:
            normalized_path = Path(path_str)
            protein_id: Optional[str] = None
            sequence_parts: List[str] = []
            try:
                with open(normalized_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        if line.startswith('>'):
                            if protein_id and sequence_parts:
                                yield protein_id, "".join(sequence_parts)
                            header = line[1:]
                            protein_id = FastaUtils.extract_id_from_header(header)
                            sequence_parts = []
                        elif protein_id is not None:
                            sequence_parts.append(line.upper())
                if protein_id and sequence_parts:
                    yield protein_id, "".join(sequence_parts)
            except FileNotFoundError:
                print(f"Error: FASTA file not found at {normalized_path}")
            except Exception as e:
                print(f"Error parsing FASTA file {normalized_path}: {e}")

    class FastaCorpus:
        """A memory-efficient corpus for Word2Vec that reads from FASTA files."""

        def __init__(self, fasta_files: List[Union[str, Path]]):
            self.fasta_files = [Path(f) for f in fasta_files]

        def __iter__(self) -> Iterator[List[str]]:
            for f_path in self.fasta_files:
                for _, sequence in FastaUtils.parse_sequences([f_path]):
                    if sequence: yield list(sequence)


# ==============================================================================
# 3. Interaction Data Loading
# ==============================================================================
class GroundTruthLoader:
    """
    Handles loading and processing of protein interaction data from files.
    """

    @staticmethod
    def get_required_ids_from_files(file_paths: List[Union[str, Path]]) -> Set[str]:
        """
        Memory-efficiently reads interaction files to get the set of all unique protein IDs.
        Reads files line-by-line to avoid loading everything into memory.
        """
        print("Gathering all required protein IDs from interaction files...")
        required_ids: Set[str] = set()
        for filepath in file_paths:
            filepath = Path(filepath)
            if not filepath.exists():
                print(f"Warning: File not found during ID gathering: {filepath}")
                continue
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in tqdm(f, desc=f"Scanning {filepath.name} for IDs", leave=False):
                        parts = [p.strip() for p in line.strip().replace('"', '').split(',')]
                        if len(parts) < 2:
                            parts = [p.strip() for p in line.strip().replace('"', '').split('\t')]
                        if len(parts) >= 2:
                            p1, p2 = parts[0], parts[1]
                            if p1: required_ids.add(p1)
                            if p2: required_ids.add(p2)
            except Exception as e:
                print(f"Error reading file {filepath} during ID gathering: {e}")
        print(f"Found {len(required_ids)} unique protein IDs across all interaction files.")
        return required_ids

    @staticmethod
    def load_interaction_pairs(filepath: Union[str, Path], label: int, sample_n: Optional[int] = None,
                               random_state: Optional[int] = None) -> List[Tuple[str, str, int]]:
        """
        Loads interaction pairs from a CSV/TSV file. Includes option for sampling.
        """
        filepath = Path(filepath)
        sampling_info = f" (sampling up to {sample_n} pairs)" if sample_n is not None else ""
        print(f"Loading pairs from: {filepath.name} (label: {label}){sampling_info}...")
        if not filepath.exists():
            print(f"Warning: Interaction file not found: {filepath}")
            return []
        try:
            # Sniff the delimiter for robustness instead of nested try-except
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline()
                sep = '\t' if '\t' in first_line else ','

            df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str, on_bad_lines='warn',
                             sep=sep)
            df.dropna(subset=['protein1', 'protein2'], inplace=True)
            df['protein1'] = df['protein1'].astype(str).str.strip()
            df['protein2'] = df['protein2'].astype(str).str.strip()
            df = df[(df['protein1'] != "") & (df['protein2'] != "")]

            if sample_n is not None and 0 < sample_n < len(df):
                df = df.sample(n=sample_n, random_state=random_state)

            pairs = [(row.protein1, row.protein2, label) for _, row in df.iterrows()]
            print(f"Successfully loaded {len(pairs)} pairs.")
            return pairs
        except Exception as e:
            print(f"Error loading interaction file {filepath}: {e}")
            return []

    @staticmethod
    def stream_interaction_pairs(filepath: Union[str, Path], label: int, batch_size: int, sample_n: Optional[int] = None,
                                 random_state: Optional[int] = None) -> Iterator[List[Tuple[str, str, int]]]:
        """
        Reads interaction pairs from a CSV/TSV file line by line and yields them in batches.
        """
        filepath = Path(filepath)
        streaming_info = f" (sampling up to {sample_n} pairs)" if sample_n is not None else ""
        print(f"Streaming pairs from: {filepath.name} (label: {label}, batch_size: {batch_size}){streaming_info}...")
        if not filepath.exists():
            print(f"Warning: Interaction file not found: {filepath}")
            return

        lines_to_read_indices: Optional[Set[int]] = None
        if sample_n is not None:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    total_lines = sum(1 for _ in f)
                if 0 < sample_n < total_lines:
                    rng = np.random.default_rng(random_state)
                    lines_to_read_indices = set(rng.choice(total_lines, sample_n, replace=False))
            except Exception as e:
                print(f"Error during pre-sampling count for {filepath}: {e}. Proceeding without sampling if possible.")

        batch: List[Tuple[str, str, int]] = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if lines_to_read_indices is not None and i not in lines_to_read_indices:
                        continue
                    parts = [p.strip() for p in line.strip().replace('"', '').split(',')]
                    if len(parts) < 2:
                        parts = [p.strip() for p in line.strip().replace('"', '').split('\t')]
                    if len(parts) >= 2:
                        p1, p2 = parts[0], parts[1]
                        if p1 and p2:
                            batch.append((p1, p2, label))
                            if len(batch) == batch_size:
                                yield batch
                                batch = []
        except Exception as e:
            print(f"Error streaming interaction file {filepath}: {e}")

        if batch:
            yield batch


# ==============================================================================
# 3. Protein ID Mapping Utilities
# ==============================================================================

# --- 3a. ID Map Generator ---
class IDMapGenerator:
    """
    Handles the complex process of generating protein ID mappings from various
    sources (API, regex, or large mapping files).
    """

    def __init__(self, config: Config):
        self.config = config
        self.fasta_files_for_mapping = config.SEQUENCE_FILE_PATHS
        self.mapping_output_file = str(config.ID_MAPPING_PATH)
        self.api_from_db = config.API_MAPPING_FROM_DB
        self.api_to_db = config.API_MAPPING_TO_DB
        self.random_seed_for_mapping = config.RANDOM_STATE
        self.mapping_mode = config.ID_MAPPING_MODE
        self.api_sample_size: Optional[int] = getattr(config, 'API_MAPPING_SAMPLE_SIZE', None)

    def generate_id_maps(self) -> Optional[Union[Mapping[str, str], 'IDMapper']]:
        """
        Main entry point for generating ID mappings.
        Returns a dictionary for 'regex'/'api' modes or an IDMapper object for 'file' mode.
        """
        if not self.mapping_output_file and self.mapping_mode != 'file':
            return None

        if self.mapping_mode == 'file':
            DataUtils.print_header("Loading Protein ID Mapping from File")
            db_path = self._create_or_get_mapping_db()
            return IDMapper(db_path) if db_path else None

        DataUtils.print_header("Generating Protein ID Mapping")
        output_dir = os.path.dirname(self.mapping_output_file)
        if output_dir: os.makedirs(output_dir, exist_ok=True)

        id_map: Dict[str, str] = {}
        if self.mapping_mode == 'api':
            id_map = self._perform_api_mapping()
        elif self.mapping_mode == 'regex':
            id_map = self._perform_regex_mapping()
        elif self.mapping_mode == 'none':
            return {}
        else:
            print(f"Warning: Unknown ID_MAPPING_MODE '{self.mapping_mode}'.")
            return {}

        if id_map:
            try:
                with open(self.mapping_output_file, 'w', encoding='utf-8') as f:
                    for original, mapped in id_map.items():
                        f.write(f"{original}\t{mapped}\n")
                print(f"ID mapping saved to {self.mapping_output_file}")
            except IOError as e:
                print(f"ERROR: Could not write ID mapping file: {e}")
        print("--- Protein ID Mapping Finished ---")
        return id_map

    def _extract_candidate_ids_from_fasta_for_mapping(self) -> Set[str]:
        if not self.fasta_files_for_mapping: return set()
        print(f"Extracting candidate IDs from: {[p.name for p in self.fasta_files_for_mapping]}...")
        # --- REFACTOR: Use the centralized, more robust FastaUtils parser ---
        # This ensures that ID extraction logic is perfectly consistent across the entire project.
        candidate_ids = {
            protein_id
            for _, (protein_id, _) in tqdm(enumerate(FastaUtils.parse_sequences(self.fasta_files_for_mapping)),
                                           desc="Scanning FASTA for IDs")
        }
        print(f"  Found {len(candidate_ids)} unique candidate IDs for API mapping.")
        return candidate_ids

    def _perform_api_mapping(self) -> Dict[str, str]:
        if not all([self.api_from_db, self.api_to_db, self.random_seed_for_mapping is not None]): return {}
        all_candidate_ids = list(self._extract_candidate_ids_from_fasta_for_mapping())

        if not all_candidate_ids: return {}
        ids_to_process = all_candidate_ids
        if self.api_sample_size is not None and 0 < self.api_sample_size < len(all_candidate_ids):
            random.seed(self.random_seed_for_mapping)
            ids_to_process = random.sample(all_candidate_ids, self.api_sample_size)
        processed_mappings: Dict[str, str] = {}
        # --- ANTICIPATORY DEBUGGING: Implement robust API polling with backoff and timeout. ---
        max_wait_time = 300  # 5 minutes total timeout per batch
        initial_sleep = 2
        for i in range(0, len(ids_to_process), 500):
            batch_ids = ids_to_process[i:i + 500]
            print(f"\nProcessing batch {i // 500 + 1}/{(len(ids_to_process) + 499) // 500}...")
            try:
                job_id = self._submit_id_mapping_job(batch_ids, self.api_from_db, self.api_to_db)
                current_sleep = initial_sleep
                total_slept = 0
                while True:
                    time.sleep(current_sleep)
                    total_slept += current_sleep
                    status = self._check_job_status(job_id)
                    print(f"  Job {job_id} status: {status}")
                    if status == "FINISHED":
                        for entry in self._get_mapping_results(job_id):
                            from_id = entry.get("from")
                            to_data = entry.get("to")
                            to_id = to_data.get("primaryAccession") if isinstance(to_data, dict) else to_data
                            if from_id and to_id: processed_mappings[from_id] = to_id
                        break
                    elif status not in ["RUNNING", "QUEUED"] or total_slept > max_wait_time:
                        if total_slept > max_wait_time: print(f"  ERROR: Job {job_id} timed out after {max_wait_time}s.")
                        break
                    current_sleep = min(current_sleep * 2, 30) # Exponential backoff, max 30s sleep
            except Exception as e:
                print(f"  Error processing batch: {e}. Skipping.")
        return processed_mappings

    def _create_or_get_mapping_db(self) -> Optional[Path]:
        """
        Creates a SQLite database from the large TSV mapping file if it doesn't already exist.
        """
        source_tsv_path = self.config.ID_MAPPING_PATH
        if not source_tsv_path or not source_tsv_path.exists():
            print(f"ERROR: Mapping file not found at {source_tsv_path}. Cannot create DB.")
            print("Ensure the file exists. It can be downloaded by setting 'ID_MAPPING_TSV' in DATA_SOURCES.")
            return None

        db_path = source_tsv_path.with_suffix('.sqlite')
        if db_path.exists():
            print(f"  Found existing ID mapping database: {db_path.name}")
            return db_path
        try:
            print(f"  Database not found. Creating new ID mapping database from {source_tsv_path.name}...")
            column_names = ['UniProtKB-AC', 'ID_type', 'ID']
            target_db = self.config.API_MAPPING_FROM_DB
            print(f"  Filtering for ID type: '{target_db}'")
            ddf = dd.read_csv(
                source_tsv_path, sep='\t', header=None, names=column_names,
                usecols=[0, 1, 2], dtype={'ID_type': 'category', 'ID': 'object', 'UniProtKB-AC': 'object'},
                blocksize='256MB'
            )
            filtered_ddf = ddf[ddf['ID_type'] == target_db]
            with sqlite3.connect(db_path) as conn, ProgressBar(dt=5.0):
                print("  Writing to database from Dask partitions (this may take a while)...")
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE id_map (original_id TEXT PRIMARY KEY, mapped_id TEXT NOT NULL)")
                for partition in filtered_ddf.partitions:
                    chunk_df = partition.compute()
                    data_to_insert = list(zip(chunk_df['ID'], chunk_df['UniProtKB-AC']))
                    if data_to_insert:
                        cursor.executemany("INSERT OR IGNORE INTO id_map (original_id, mapped_id) VALUES (?, ?)",
                                           data_to_insert)
                conn.commit()
            print(f"  Successfully created ID mapping database: {db_path.name}")
            return db_path
        except Exception as e:
            print(f"  An error occurred while creating the mapping database: {e}")
            if 'db_path' in locals() and db_path.exists():
                db_path.unlink()
            return None

    def _perform_regex_mapping(self) -> Dict[str, str]:
        if not self.fasta_files_for_mapping: return {}
        print(f"Starting Regex ID mapping for: {[p.name for p in self.fasta_files_for_mapping]}...")
        id_map = {}
        for fasta_file in self.fasta_files_for_mapping:
            try:
                for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc=f"Parsing {fasta_file.name} with Regex"):
                    _, canonical_id = self._extract_canonical_id_and_type_from_header(record.description)
                    if canonical_id:
                        id_map[record.id] = canonical_id
                        first_word = record.description.split()[0]
                        if first_word != record.id: id_map[first_word] = canonical_id
            except Exception as e:
                print(f"An error during regex mapping on {fasta_file}: {e}")
        print(f"Regex mapping complete. Found {len(id_map)} potential mappings.")
        return id_map

    @staticmethod
    def _submit_id_mapping_job(ids_to_map: List[str], from_db: str, to_db: str) -> str:
        response = requests.post("https://rest.uniprot.org/idmapping/run",
                                 data={"ids": ",".join(ids_to_map), "from": from_db, "to": to_db})
        response.raise_for_status()
        job_id = response.json().get("jobId")
        if not job_id: raise ValueError("Failed to submit job to UniProt.")
        print(f"  UniProt API job submitted for {len(ids_to_map)} IDs. Job ID: {job_id}")
        return job_id

    @staticmethod
    def _check_job_status(job_id: str) -> str:
        response = requests.get(f"https://rest.uniprot.org/idmapping/status/{job_id}")
        response.raise_for_status()
        return response.json().get("jobStatus", "UNKNOWN")

    @staticmethod
    def _get_mapping_results(job_id: str) -> List[Dict]:
        response = requests.get(f"https://rest.uniprot.org/idmapping/results/{job_id}?format=json")
        response.raise_for_status()
        return response.json().get("results", [])

    @staticmethod
    def _extract_canonical_id_and_type_from_header(header: str) -> Tuple[Optional[str], Optional[str]]:
        hid = header.strip().lstrip('>')
        up_match = re.match(r"^(?:sp|tr)\|([OPQ]?[A-Z0-9]{5,9}(?:-\d+)?)\|", hid, re.IGNORECASE)
        if up_match: return "UniProt", up_match.group(1)
        uniref_match = re.match(r"^(UniRef\d{2,3})_([A-Z0-9]+)", hid, re.IGNORECASE)
        if uniref_match: return "UniProt (from UniRef)", uniref_match.group(2)
        plain_match_strict = re.match(r"^([OPQ]?[A-Z0-9]{5,9}(?:-\d+)?)", hid.split()[0])
        if plain_match_strict: return "UniProt (assumed)", plain_match_strict.group(1)
        return "Unknown", hid.split()[0]


# --- 3b. Memory-Efficient ID Map Reader ---
class IDMapper:
    """
    A memory-efficient wrapper for a SQLite database that provides a dictionary-like
    lookup for protein ID mappings. This avoids loading millions of mappings into RAM.
    Should be used with a 'with' statement to ensure the database connection is managed.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        if not self.db_path.exists():
            raise FileNotFoundError(f"ID Mapping database not found at {self.db_path}")

    def __enter__(self):
        # Connect to the DB in read-only mode for safety and in WAL mode for better read performance.
        self.conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Fetches the mapped ID for a given original ID. Implements the dict.get() interface."""
        if not self.conn:
            raise ConnectionError("Database connection is not open. Use this object within a 'with' block.")
        cursor = self.conn.cursor()
        cursor.execute("SELECT mapped_id FROM id_map WHERE original_id = ?", (key,))
        result = cursor.fetchone()
        return result[0] if result else default


# ==============================================================================
# 4. Dask Helpers for ProtGram Graph Builder
# ==============================================================================
class ProtgramDaskHelpers:
    """
    Contains static helper methods used exclusively by the Dask pipeline
    in the GraphBuilder class. Isolating them here cleans up the namespace.
    """

    @staticmethod
    def _preprocess_sequence_tuple_for_bag(seq_tuple: Tuple[str, str], add_initial_space: bool) -> Tuple[str, str]:
        """Prepares a sequence tuple for Dask Bag processing."""
        pid, seq_text = seq_tuple
        # Add space padding for consistent n-gram extraction at sequence boundaries
        modified_seq_text = f" {seq_text}" if add_initial_space else str(seq_text)
        return pid, f"{modified_seq_text} "

    @staticmethod
    def _extract_ngrams_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int) -> Iterator[str]:
        """Extracts n-grams from a single processed sequence."""
        _, processed_seq_text = seq_tuple
        if len(processed_seq_text) >= n_val:
            for i in range(len(processed_seq_text) - n_val + 1):
                yield processed_seq_text[i:i + n_val]

    @staticmethod
    def _extract_edges_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int,
                                           ngram_to_id_map: Dict[str, int]) -> Iterator[str]:
        """Extracts n-gram transitions (edges) from a single processed sequence."""
        _, processed_seq_text = seq_tuple
        if len(processed_seq_text) >= n_val + 1:
            for i in range(len(processed_seq_text) - n_val):
                source_id = ngram_to_id_map.get(processed_seq_text[i:i + n_val])
                target_id = ngram_to_id_map.get(processed_seq_text[i + 1:i + 1 + n_val])
                if source_id is not None and target_id is not None:
                    # Yield a string representation for easy writing to text files
                    yield f"{source_id} {target_id}"

def prepare_pyg_data_from_protgram_graph(model_type: str, graph: 'DirectedNgramGraph', features: torch.Tensor,
                                         labels: Optional[torch.Tensor], use_homo_hetero_paths: bool) -> Data:
    """
    A centralized utility to prepare a PyG Data object from a DirectedNgramGraph,
    tailored to the specific model's needs. This eliminates duplicated logic
    between the ProtGram and Singleton trainers.
    """
    data_dict: Dict[str, Any] = {'x': features, 'y': labels, 'graph_obj': graph}
    model_name_lower = model_type.lower()

    if model_name_lower == 'directgcn':
        # --- FIX: Pass the correctly processed matrices to the model ---
        # The undirected path, and the specialized mathcal_A for directed paths.
        data_dict.update({
            'edge_index_undirected_norm': graph.A_undirected_norm_sparse.indices(),
            'edge_weight_undirected_norm': graph.A_undirected_norm_sparse.values(),
            'edge_index_mathcal_in': graph.mathcal_A_in.indices(),
            'edge_weight_mathcal_in': graph.mathcal_A_in.values(),
            'edge_index_mathcal_out': graph.mathcal_A_out.indices(),
            'edge_weight_mathcal_out': graph.mathcal_A_out.values()
        })
        # Conditionally add the NORMALIZED homophily/heterophily paths.
        if use_homo_hetero_paths and graph.A_homo_w is not None and graph.A_hetero_w is not None:
            print("  Preparing data with normalized homophily/heterophily paths...")
            data_dict.update({
                'edge_index_homo_norm': graph.A_homo_norm.indices(), 'edge_weight_homo_norm': graph.A_homo_norm.values(),
                'edge_index_hetero_norm': graph.A_hetero_norm.indices(), 'edge_weight_hetero_norm': graph.A_hetero_norm.values()
            })

    elif model_name_lower == 'rgcn':
        # RGCN requires a single edge_index and an edge_type tensor.
        edge_index_out = graph.A_out_w.indices()
        edge_index_in = graph.A_in_w.indices()
        device = edge_index_out.device
        edge_type_out = torch.zeros(edge_index_out.size(1), dtype=torch.long, device=device)
        edge_type_in = torch.ones(edge_index_in.size(1), dtype=torch.long, device=device)
        data_dict['edge_index'] = torch.cat([edge_index_out, edge_index_in], dim=1)
        data_dict['edge_type'] = torch.cat([edge_type_out, edge_type_in])

    elif model_name_lower == 'dirgnn':
        # TongDiGCN requires separate forward and backward edge indices.
        data_dict['edge_index'] = graph.A_out_w.indices()
        # --- FIX: Add the corresponding edge weights for the forward GCN pass ---
        data_dict['edge_attr'] = graph.A_out_w.values()
        data_dict['edge_index_backward'] = graph.A_in_w.indices()

    else:  # Default for standard GNNs (GCN, GAT, GraphSAGE, etc.)
        # These models expect a single, undirected, weighted graph.
        print(f"  Preparing data for standard GNN '{model_type}' using undirected normalized graph.")
        data_dict['edge_index'] = graph.A_undirected_norm_sparse.indices()
        data_dict['edge_attr'] = graph.A_undirected_norm_sparse.values()

    return Data.from_dict(data_dict)
