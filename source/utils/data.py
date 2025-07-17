# ==============================================================================
# MODULE: utils/data.py
# PURPOSE: Contains all data loading utilities for the PPI trainers,
#          including FASTA parsing, ID mapping, and interaction data loading.
# VERSION: 2.3 (Introduced on-disk SQLite DB for memory-safe ID mapping)
# AUTHOR: Islam Ebeid (Refactored by Coding Partner)
# ==============================================================================

import os
import pickle
import random
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union

import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
import requests  # For DataLoader ID mapping
from Bio import SeqIO
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

from configuration.config import Config


# ==============================================================================
# --- NEW: Memory-Efficient ID Mapper ---
# ==============================================================================
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
# --- Ground Truth and Interaction Data Loading ---
# ==============================================================================
class GroundTruthLoader:
    """
    Handles loading and processing of protein interaction data from files.
    """

    @staticmethod
    def get_required_ids_from_files(file_paths: List[str]) -> Set[str]:
        """
        Memory-efficiently reads interaction files to get the set of all unique protein IDs.
        Reads files line-by-line to avoid loading everything into memory.
        """
        print("Gathering all required protein IDs from interaction files...")
        required_ids: Set[str] = set()
        for filepath in file_paths:
            filepath = os.path.normpath(filepath)
            if not os.path.exists(filepath):
                print(f"Warning: File not found during ID gathering: {filepath}")
                continue
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in tqdm(f, desc=f"Scanning {os.path.basename(filepath)} for IDs", leave=False):
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
    def load_interaction_pairs(filepath: str, label: int, sample_n: Optional[int] = None, random_state: Optional[int] = None) -> List[Tuple[str, str, int]]:
        """
        Loads interaction pairs from a CSV/TSV file. Includes option for sampling.
        """
        filepath = os.path.normpath(filepath)
        sampling_info = f" (sampling up to {sample_n} pairs)" if sample_n is not None else ""
        print(f"Loading pairs from: {os.path.basename(filepath)} (label: {label}){sampling_info}...")
        if not os.path.exists(filepath):
            print(f"Warning: Interaction file not found: {filepath}")
            return []
        try:
            try:
                df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str, on_bad_lines='warn', sep=',')
                if df.shape[1] < 2 and os.path.getsize(filepath) > 0:
                    df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str, on_bad_lines='warn', sep='\t')
            except pd.errors.ParserError:
                df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str, on_bad_lines='warn', sep='\t')

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
    def stream_interaction_pairs(filepath: str, label: int, batch_size: int, sample_n: Optional[int] = None, random_state: Optional[int] = None) -> Iterator[List[Tuple[str, str, int]]]:
        """
        Reads interaction pairs from a CSV/TSV file line by line and yields them in batches.
        """
        filepath = os.path.normpath(filepath)
        streaming_info = f" (sampling up to {sample_n} pairs)" if sample_n is not None else ""
        print(f"Streaming pairs from: {os.path.basename(filepath)} (label: {label}, batch_size: {batch_size}){streaming_info}...")
        if not os.path.exists(filepath):
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
# --- Sequence Parsing and ID Mapping ---
# ==============================================================================
class DataLoader:
    """
    Utilities for parsing FASTA files, mapping protein identifiers, and providing FASTA corpus.
    If ID mapping is required, an instance should be created with a Config object.
    The `parse_sequences` method and `_FastaCorpus` can be used statically/nested.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config
        if config:
            self.fasta_files_for_mapping = config.SEQUENCE_FILE_PATHS
            self.mapping_output_file = str(config.ID_MAPPING_PATH)
            self.api_from_db = config.API_MAPPING_FROM_DB
            self.api_to_db = config.API_MAPPING_TO_DB
            self.random_seed_for_mapping = config.RANDOM_STATE
            self.mapping_mode = config.ID_MAPPING_MODE
            self.api_sample_size: Optional[int] = getattr(config, 'API_MAPPING_SAMPLE_SIZE', None)
        else:
            self.fasta_files_for_mapping, self.mapping_output_file, self.api_from_db = [], None, None
            self.api_to_db, self.random_seed_for_mapping, self.mapping_mode = None, None, 'none'
            self.api_sample_size = None

    @staticmethod
    def parse_sequences(fasta_filepaths: List[str]) -> Iterator[Tuple[str, str]]:
        """
        An efficient FASTA parser that reads one or more FASTA files, yielding an ID and sequence for each record.
        """
        for path_str in fasta_filepaths:
            normalized_path = os.path.normpath(path_str)
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
                            parts = header.split('|')
                            protein_id = parts[1] if len(parts) > 1 and parts[1] else header.split()[0]
                            sequence_parts = []
                        elif protein_id is not None:
                            sequence_parts.append(line.upper())
                if protein_id and sequence_parts:
                    yield protein_id, "".join(sequence_parts)
            except FileNotFoundError:
                print(f"Error: FASTA file not found at {normalized_path}")
            except Exception as e:
                print(f"Error parsing FASTA file {normalized_path}: {e}")

    class _FastaCorpus:
        """A memory-efficient corpus for Word2Vec that reads from FASTA files."""
        def __init__(self, fasta_files: List[str]):
            self.fasta_files = [os.path.normpath(f) for f in fasta_files]
        def __iter__(self) -> Iterator[List[str]]:
            for f_path in self.fasta_files:
                for _, sequence in DataLoader.parse_sequences([f_path]):
                    if sequence: yield list(sequence)

    def _extract_candidate_ids_from_fasta_for_mapping(self) -> Set[str]:
        if not self.fasta_files_for_mapping: return set()
        candidate_ids = set()
        print(f"Extracting candidate IDs from: {[p.name for p in self.fasta_files_for_mapping]}...")
        for fasta_file in self.fasta_files_for_mapping:
            try:
                with open(fasta_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in tqdm(f, desc=f"Scanning {fasta_file.name}", leave=False):
                        if line.startswith('>'):
                            header = line[1:].strip()
                            parts = header.split('|')
                            candidate_ids.add(parts[1].strip() if len(parts) > 1 and parts[1] else header.split()[0].strip())
            except Exception as e: print(f"ERROR reading FASTA {fasta_file}: {e}")
        print(f"Found {len(candidate_ids)} unique candidate IDs for API mapping.")
        return candidate_ids

    @staticmethod
    def _submit_id_mapping_job(ids_to_map: List[str], from_db: str, to_db: str) -> str:
        response = requests.post("https://rest.uniprot.org/idmapping/run", data={"ids": ",".join(ids_to_map), "from": from_db, "to": to_db})
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

    def _perform_api_mapping(self) -> Dict[str, str]:
        if not all([self.api_from_db, self.api_to_db, self.random_seed_for_mapping is not None]): return {}
        all_candidate_ids = list(self._extract_candidate_ids_from_fasta_for_mapping())
        if not all_candidate_ids: return {}
        ids_to_process = all_candidate_ids
        if self.api_sample_size is not None and 0 < self.api_sample_size < len(all_candidate_ids):
            random.seed(self.random_seed_for_mapping)
            ids_to_process = random.sample(all_candidate_ids, self.api_sample_size)
        processed_mappings: Dict[str, str] = {}
        for i in range(0, len(ids_to_process), 500):
            batch_ids = ids_to_process[i:i + 500]
            print(f"\nProcessing batch {i // 500 + 1}/{(len(ids_to_process) + 499) // 500}...")
            try:
                job_id = self._submit_id_mapping_job(batch_ids, self.api_from_db, self.api_to_db)
                while True:
                    time.sleep(2)
                    status = self._check_job_status(job_id)
                    print(f"  Job {job_id} status: {status}")
                    if status == "FINISHED":
                        for entry in self._get_mapping_results(job_id):
                            from_id = entry.get("from")
                            to_data = entry.get("to")
                            to_id = to_data.get("primaryAccession") if isinstance(to_data, dict) else to_data
                            if from_id and to_id: processed_mappings[from_id] = to_id
                        break
                    elif status not in ["RUNNING", "QUEUED"]: break
            except Exception as e: print(f"  Error processing batch: {e}. Skipping.")
        return processed_mappings

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

    def _create_or_get_mapping_db(self) -> Optional[Path]:
        """
        Creates a SQLite database from the large TSV mapping file if it doesn't already exist.
        This is the core fix for the OOM error, as it processes the file in chunks
        and writes directly to a database on disk, avoiding loading the full map into RAM.
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
                        cursor.executemany("INSERT OR IGNORE INTO id_map (original_id, mapped_id) VALUES (?, ?)", data_to_insert)
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
            except Exception as e: print(f"An error during regex mapping on {fasta_file}: {e}")
        print(f"Regex mapping complete. Found {len(id_map)} potential mappings.")
        return id_map

    def generate_id_maps(self) -> Optional[Union[Mapping[str, str], IDMapper]]:
        """
        Main entry point for generating ID mappings.
        Returns a dictionary for 'regex'/'api' modes or an IDMapper object for 'file' mode.
        """
        if not self.config: return None
        if not self.mapping_output_file and self.mapping_mode != 'file': return None

        if self.mapping_mode == 'file':
            DataUtils.print_header("Loading Protein ID Mapping from File")
            db_path = self._create_or_get_mapping_db()
            return IDMapper(db_path) if db_path else None

        DataUtils.print_header("Generating Protein ID Mapping")
        output_dir = os.path.dirname(self.mapping_output_file)
        if output_dir: os.makedirs(output_dir, exist_ok=True)
        id_map: Dict[str, str] = {}
        if self.mapping_mode == 'api': id_map = self._perform_api_mapping()
        elif self.mapping_mode == 'regex': id_map = self._perform_regex_mapping()
        elif self.mapping_mode == 'none': return {}
        else: print(f"Warning: Unknown ID_MAPPING_MODE '{self.mapping_mode}'."); return {}

        if id_map:
            try:
                with open(self.mapping_output_file, 'w', encoding='utf-8') as f:
                    for original, mapped in id_map.items(): f.write(f"{original}\t{mapped}\n")
                print(f"ID mapping saved to {self.mapping_output_file}")
            except IOError as e: print(f"ERROR: Could not write ID mapping file: {e}")
        print("--- Protein ID Mapping Finished ---")
        return id_map

    @staticmethod
    def _preprocess_sequence_tuple_for_bag(seq_tuple: Tuple[str, str], add_initial_space: bool) -> Tuple[str, str]:
        pid, seq_text = seq_tuple
        modified_seq_text = f" {seq_text}" if add_initial_space else str(seq_text)
        return pid, f"{modified_seq_text} "

    @staticmethod
    def _extract_ngrams_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int) -> Iterator[str]:
        _, processed_seq_text = seq_tuple
        if len(processed_seq_text) >= n_val:
            for i in range(len(processed_seq_text) - n_val + 1):
                yield processed_seq_text[i:i + n_val]

    @staticmethod
    def _extract_edges_from_sequence_tuple(seq_tuple: Tuple[str, str], n_val: int, ngram_to_id_map: Dict[str, int]) -> Iterator[str]:
        _, processed_seq_text = seq_tuple
        if len(processed_seq_text) >= n_val + 1:
            for i in range(len(processed_seq_text) - n_val):
                source_id = ngram_to_id_map.get(processed_seq_text[i:i + n_val])
                target_id = ngram_to_id_map.get(processed_seq_text[i + 1:i + 1 + n_val])
                if source_id is not None and target_id is not None:
                    yield f"{source_id} {target_id}\n"


# ==============================================================================
# --- General Data Utilities ---
# ==============================================================================
class DataUtils:
    """General data utility functions."""

    @staticmethod
    def print_header(title: str):
        border = "=" * (len(title) + 6)
        print(f"\n{border}\n### {title} ###\n{border}\n")

    @staticmethod
    def save_object(obj: any, filepath: str):
        filepath = os.path.normpath(filepath)
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f: pickle.dump(obj, f)
            print(f"Object saved to {filepath}")
        except Exception as e: print(f"Error saving object to {filepath}: {e}")

    @staticmethod
    def load_object(filepath: str) -> Optional[any]:
        filepath = os.path.normpath(filepath)
        if not os.path.exists(filepath): return None
        try:
            with open(filepath, 'rb') as f: return pickle.load(f)
        except Exception as e: print(f"Error loading object from {filepath}: {e}"); return None

    @staticmethod
    def save_dataframe_to_csv(df: pd.DataFrame, output_path: str, index: bool = False):
        output_path = os.path.normpath(output_path)
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=index)
            print(f"DataFrame saved to: {output_path}")
        except Exception as e: print(f"Error saving DataFrame to {output_path}: {e}")

    @staticmethod
    def check_h5_embeddings_integrity(h5_filepath: str, num_samples_to_check: int = 5):
        h5_filepath = os.path.normpath(h5_filepath)
        DataUtils.print_header(f"Checking HDF5 file: {os.path.basename(h5_filepath)}")
        if not os.path.exists(h5_filepath) or not h5py.is_hdf5(h5_filepath):
            print(f"Error: File at '{h5_filepath}' is not a valid HDF5 file or does not exist.")
            return
        try:
            with h5py.File(h5_filepath, 'r') as hf:
                keys = list(hf.keys())
                if not keys: print("HDF5 check: File is empty."); return
                print(f"Found {len(keys)} total embeddings. Inspecting up to {num_samples_to_check} samples:")
                sample_keys = random.sample(keys, min(len(keys), num_samples_to_check))
                for i, key in enumerate(sample_keys):
                    dataset = hf.get(key)
                    if not isinstance(dataset, h5py.Dataset): continue
                    emb = dataset[:]
                    print(f"  - Sample {i + 1}: Key='{key}', Shape={emb.shape}, DType={emb.dtype}")
                    if np.isnan(emb).any(): print("    - WARNING: Embedding contains NaN values.")
                    if np.isinf(emb).any(): print("    - WARNING: Embedding contains Inf values.")
        except Exception as e: print(f"An error occurred while checking HDF5 file '{h5_filepath}': {e}")