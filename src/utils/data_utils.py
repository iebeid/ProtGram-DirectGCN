# G:/My Drive/Knowledge/Research/TWU/Topics/AI in Proteomics/Protein-protein interaction prediction/Code/ProtDiGCN/src/utils/data_utils.py
# ==============================================================================
# MODULE: utils/data_utils.py
# PURPOSE: Contains all data loading utilities for the PPI pipeline,
#          including FASTA parsing, ID mapping, and interaction data loading.
# VERSION: 2.2 (Integrated _FastaCorpus into DataLoader)
# ==============================================================================

import os
import pickle
import random
import re  # For DataLoader ID mapping
import time  # For DataLoader ID mapping
from typing import List, Optional, Dict, Set, Tuple, Iterator

import h5py
import numpy as np
import pandas as pd
import requests  # For DataLoader ID mapping
from Bio import SeqIO  # For DataLoader ID mapping
from tqdm.auto import tqdm

# Assuming your Config class is in src.config
# This import is for type hinting and accessing config values.
from src.config import Config


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
                        # Attempt to split by comma, then by tab if comma fails
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
            # Try to infer separator, default to comma
            try:
                df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str, on_bad_lines='warn', sep=',')
                if df.shape[1] < 2 and os.path.getsize(filepath) > 0:  # Check if comma was a good separator
                    df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str, on_bad_lines='warn', sep='\t')
            except pd.errors.ParserError:  # Fallback to tab if comma parsing fails badly
                df = pd.read_csv(filepath, header=None, names=['protein1', 'protein2'], dtype=str, on_bad_lines='warn', sep='\t')

            df.dropna(subset=['protein1', 'protein2'], inplace=True)  # Ensure both protein IDs are present
            df['protein1'] = df['protein1'].astype(str).str.strip()
            df['protein2'] = df['protein2'].astype(str).str.strip()
            df = df[(df['protein1'] != "") & (df['protein2'] != "")]  # Filter out empty strings after stripping

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
                        if p1 and p2:  # Ensure both IDs are non-empty
                            batch.append((p1, p2, label))
                            if len(batch) == batch_size:
                                yield batch
                                batch = []
        except Exception as e:
            print(f"Error streaming interaction file {filepath}: {e}")

        if batch:  # Yield the last batch if it's not empty
            yield batch


class DataLoader:
    """
    Utilities for parsing FASTA files, mapping protein identifiers, and providing FASTA corpus.
    If ID mapping is required, an instance should be created with a Config object.
    The `parse_sequences` method and `_FastaCorpus` can be used statically/nested.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initializes the DataLoader.

        Args:
            config (Optional[Config]): The main configuration object.
                                       Required if ID mapping functionalities are to be used.
        """
        self.config = config
        if config:
            # Attributes for ID mapping
            self.fasta_path_for_mapping = str(config.GCN_INPUT_FASTA_PATH)
            self.mapping_output_file = str(config.ID_MAPPING_OUTPUT_FILE)
            self.api_from_db = config.API_MAPPING_FROM_DB
            self.api_to_db = config.API_MAPPING_TO_DB
            self.random_seed_for_mapping = config.RANDOM_STATE
            self.mapping_mode = config.ID_MAPPING_MODE
            self.api_sample_size: Optional[int] = getattr(config, 'API_MAPPING_SAMPLE_SIZE', None)
        else:
            self.fasta_path_for_mapping = None
            self.mapping_output_file = None
            self.api_from_db = None
            self.api_to_db = None
            self.random_seed_for_mapping = None
            self.mapping_mode = 'none'  # Default to no mapping if no config
            self.api_sample_size = None

    @staticmethod
    def parse_sequences(fasta_filepath: str) -> Iterator[Tuple[str, str]]:
        """
        An efficient FASTA parser that reads one sequence at a time, yielding an ID and sequence.
        """
        fasta_filepath = os.path.normpath(fasta_filepath)
        protein_id: Optional[str] = None
        sequence_parts: List[str] = []
        try:
            with open(fasta_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('>'):
                        if protein_id and sequence_parts:
                            yield protein_id, "".join(sequence_parts)

                        header = line[1:]
                        parts = header.split('|')
                        if len(parts) > 1 and parts[1]:
                            protein_id = parts[1]
                        else:
                            protein_id = header.split()[0]
                        sequence_parts = []
                    elif protein_id is not None:
                        sequence_parts.append(line.upper())
            if protein_id and sequence_parts:
                yield protein_id, "".join(sequence_parts)
        except FileNotFoundError:
            print(f"Error: FASTA file not found at {fasta_filepath}")
        except Exception as e:
            print(f"Error parsing FASTA file {fasta_filepath}: {e}")

    class _FastaCorpus:  # Moved _FastaCorpus here as a nested class
        """A memory-efficient corpus for Word2Vec that reads from FASTA files."""

        def __init__(self, fasta_files: List[str]):
            self.fasta_files = [os.path.normpath(f) for f in fasta_files]

        def __iter__(self) -> Iterator[List[str]]:
            for f_path in self.fasta_files:
                # Use the outer DataLoader's static parse_sequences method
                for _, sequence in DataLoader.parse_sequences(f_path):
                    if sequence:  # Ensure sequence is not empty
                        yield list(sequence)

    # --- ID Mapping Methods (adapted from ProteinIDMapper) ---
    # These methods remain the same as in the previous version where ProteinIDMapper was consolidated here.
    # For brevity, I'm not repeating them all, but they should be present here.

    def _extract_candidate_ids_from_fasta_for_mapping(self) -> Set[str]:
        if not self.fasta_path_for_mapping:
            print("ERROR: FASTA path for mapping is not configured.")
            return set()
        candidate_ids = set()
        print(f"Extracting candidate IDs from: {os.path.basename(self.fasta_path_for_mapping)}...")
        try:
            with open(self.fasta_path_for_mapping, 'r', encoding='utf-8', errors='ignore') as f:
                for line in tqdm(f, desc="Scanning FASTA headers for mapping", leave=False):
                    if line.startswith('>'):
                        header = line[1:].strip()
                        parts = header.split('|')
                        if len(parts) > 1 and parts[1]:
                            candidate_ids.add(parts[1].strip())
                            continue
                        candidate_ids.add(header.split()[0].strip())
        except FileNotFoundError:
            print(f"ERROR: FASTA file not found at {self.fasta_path_for_mapping}")
            return set()
        print(f"Found {len(candidate_ids)} unique candidate IDs for API mapping.")
        return candidate_ids

    @staticmethod
    def _submit_id_mapping_job(ids_to_map: List[str], from_db: str, to_db: str) -> str:
        payload = {"ids": ",".join(ids_to_map), "from": from_db, "to": to_db}
        response = requests.post("https://rest.uniprot.org/idmapping/run", data=payload)
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
        if not all([self.api_from_db, self.api_to_db, self.random_seed_for_mapping is not None]):
            print("ERROR: API mapping parameters (from_db, to_db, random_seed) are not fully configured.")
            return {}
        all_candidate_ids = list(self._extract_candidate_ids_from_fasta_for_mapping())
        if not all_candidate_ids:
            return {}
        ids_to_process = all_candidate_ids
        if self.api_sample_size is not None and 0 < self.api_sample_size < len(all_candidate_ids):
            print(f"Randomly sampling {self.api_sample_size} IDs from {len(all_candidate_ids)} total candidates.")
            random.seed(self.random_seed_for_mapping)
            ids_to_process = random.sample(all_candidate_ids, self.api_sample_size)
        total_ids = len(ids_to_process)
        processed_mappings: Dict[str, str] = {}
        batch_size = 500
        request_interval = 2
        print(f"\nStarting UniProt ID mapping for {total_ids} IDs (from: {self.api_from_db}, to: {self.api_to_db}) in batches of {batch_size}.")
        for i in range(0, total_ids, batch_size):
            batch_ids = ids_to_process[i:i + batch_size]
            print(f"\nProcessing batch {i // batch_size + 1}/{(total_ids + batch_size - 1) // batch_size}...")
            try:
                job_id = self._submit_id_mapping_job(batch_ids, self.api_from_db, self.api_to_db)
                while True:
                    time.sleep(request_interval)
                    status = self._check_job_status(job_id)
                    print(f"  Job {job_id} status: {status}")
                    if status == "FINISHED":
                        results = self._get_mapping_results(job_id)
                        for entry in results:
                            from_id = entry.get("from")
                            to_data = entry.get("to")
                            to_id = None
                            if isinstance(to_data, dict):
                                to_id = to_data.get("primaryAccession")
                            elif isinstance(to_data, str):
                                to_id = to_data
                            if from_id and to_id:
                                processed_mappings[from_id] = to_id
                        break
                    elif status not in ["RUNNING", "QUEUED"]:
                        print(f"  Job {job_id} failed or has an unexpected status: {status}")
                        break
            except requests.exceptions.RequestException as e_req:
                print(f"  Network error processing batch: {e_req}. Skipping batch.")
            except ValueError as e_val:
                print(f"  Value error processing batch: {e_val}. Skipping batch.")
            except Exception as e:
                print(f"  Unexpected error processing batch: {e}. Skipping batch.")
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

    def _perform_regex_mapping(self) -> Dict[str, str]:
        if not self.fasta_path_for_mapping:
            print("ERROR: FASTA path for mapping is not configured.")
            return {}
        print(f"Starting Regex ID mapping for: {os.path.basename(self.fasta_path_for_mapping)}...")
        id_map = {}
        try:
            for record in tqdm(SeqIO.parse(self.fasta_path_for_mapping, "fasta"), desc="Parsing FASTA with Regex for Mapping"):
                original_id_from_record = record.id
                full_header = record.description
                _, canonical_id = self._extract_canonical_id_and_type_from_header(full_header)
                if canonical_id and canonical_id != original_id_from_record:
                    id_map[original_id_from_record] = canonical_id
                first_word_full_header = full_header.split()[0]
                if canonical_id and first_word_full_header != canonical_id and first_word_full_header not in id_map:
                    id_map[first_word_full_header] = canonical_id
                if canonical_id and original_id_from_record != canonical_id and original_id_from_record not in id_map:
                    id_map[original_id_from_record] = canonical_id
        except FileNotFoundError:
            print(f"ERROR: FASTA file not found at {self.fasta_path_for_mapping}")
        except Exception as e:
            print(f"An error occurred during regex mapping: {e}")
        print(f"Regex mapping complete. Found {len(id_map)} potential mappings.")
        return id_map

    def generate_id_maps(self) -> Dict[str, str]:
        if not self.config:
            print("ERROR: DataLoader not initialized with a Config object. Cannot generate ID maps.")
            return {}
        if not self.mapping_output_file:
            print("ERROR: Mapping output file is not configured.")
            return {}
        DataUtils.print_header("Generating Protein ID Mapping")
        output_dir = os.path.dirname(self.mapping_output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        id_map: Dict[str, str] = {}
        if self.mapping_mode == 'api':
            print("Mode: API Mapping")
            id_map = self._perform_api_mapping()
        elif self.mapping_mode == 'regex':
            print("Mode: Regex Mapping")
            id_map = self._perform_regex_mapping()
        elif self.mapping_mode == 'none':
            print("ID mapping mode is 'none'. No mapping will be performed.")
        else:
            print(f"Warning: Unknown ID_MAPPING_MODE '{self.mapping_mode}'. No mapping performed.")
        if id_map:
            try:
                with open(self.mapping_output_file, 'w', encoding='utf-8') as f:
                    for original, mapped in id_map.items():
                        f.write(f"{original}\t{mapped}\n")
                print(f"ID mapping saved to {self.mapping_output_file}")
            except IOError as e:
                print(f"ERROR: Could not write ID mapping file to {self.mapping_output_file}: {e}")
        elif self.mapping_mode not in ['none', 'unknown']:
            print("No ID mappings were generated or an error occurred.")
        print("--- Protein ID Mapping Finished ---")
        return id_map


class DataUtils:
    """
    General data utility functions.
    """

    @staticmethod
    def print_header(title: str):
        """Prints a formatted header to the console."""
        border = "=" * (len(title) + 6)  # Adjusted for "### "
        print(f"\n{border}\n### {title} ###\n{border}\n")

    @staticmethod
    def save_object(obj: any, filepath: str):
        """Saves a Python object to a file using pickle."""
        filepath = os.path.normpath(filepath)
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(obj, f)
            print(f"Object saved to {filepath}")
        except Exception as e:
            print(f"Error saving object to {filepath}: {e}")

    @staticmethod
    def load_object(filepath: str) -> Optional[any]:
        """Loads a Python object from a pickle file."""
        filepath = os.path.normpath(filepath)
        if not os.path.exists(filepath):
            print(f"Error: File not found for loading object: {filepath}")
            return None
        try:
            with open(filepath, 'rb') as f:
                obj = pickle.load(f)
            print(f"Object loaded from {filepath}")
            return obj
        except Exception as e:
            print(f"Error loading object from {filepath}: {e}")
            return None

    @staticmethod
    def save_dataframe_to_csv(df: pd.DataFrame, output_path: str, index: bool = False):
        """Saves a pandas DataFrame to a CSV file."""
        output_path = os.path.normpath(output_path)
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=index)
            print(f"DataFrame saved to: {output_path}")
        except Exception as e:
            print(f"Error saving DataFrame to {output_path}: {e}")

    @staticmethod
    def check_h5_embeddings_integrity(h5_filepath: str, num_samples_to_check: int = 5):
        """
        Reads an HDF5 embedding file to check its integrity and properties.
        """
        h5_filepath = os.path.normpath(h5_filepath)
        DataUtils.print_header(f"Checking HDF5 file: {os.path.basename(h5_filepath)}")
        if not os.path.exists(h5_filepath):
            print(f"Error: File not found at '{h5_filepath}'")
            return
        if not h5py.is_hdf5(h5_filepath):
            print(f"Error: File at '{h5_filepath}' is not a valid HDF5 file.")
            return

        try:
            with h5py.File(h5_filepath, 'r') as hf:
                keys = list(hf.keys())
                if not keys:
                    print("HDF5 check: File is empty or contains no embeddings.")
                    return

                print(f"Found {len(keys)} total embeddings. Inspecting up to {num_samples_to_check} samples:")
                actual_samples_to_check = min(len(keys), num_samples_to_check)
                if actual_samples_to_check == 0 and len(keys) > 0:
                    print("Warning: No samples to check despite keys being present.")
                    return

                sample_keys = random.sample(keys, actual_samples_to_check) if actual_samples_to_check > 0 else []

                for i, key in enumerate(sample_keys):
                    dataset = hf.get(key)
                    if not isinstance(dataset, h5py.Dataset):
                        print(f"  - Key '{key}' is not a valid HDF5 Dataset.")
                        continue

                    emb = dataset[:]
                    print(f"  - Sample {i + 1}: Key='{key}', Shape={emb.shape}, DType={emb.dtype}")

                    if emb.ndim != 1 and emb.ndim != 2:
                        print(f"    - Note: Embedding is not a 1D or 2D vector (ndim={emb.ndim}).")
                    if np.isnan(emb).any():
                        print("    - WARNING: Embedding contains NaN values.")
                    if np.isinf(emb).any():
                        print("    - WARNING: Embedding contains Inf values.")
                    if emb.size == 0:
                        print("    - WARNING: Embedding is empty (size 0).")
        except Exception as e:
            print(f"An error occurred while checking HDF5 file '{h5_filepath}': {e}")
