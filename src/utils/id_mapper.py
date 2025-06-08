# ==============================================================================
# MODULE: utils/id_mapping.py
# PURPOSE: Contains functions for mapping protein identifiers, either locally
#          with regex or via the UniProt API.
# ==============================================================================

import requests
import time
import os
import random
import re
from typing import List, Dict, Set, Optional, Tuple
from tqdm.auto import tqdm
from Bio import SeqIO


# --- UniProt API Functions (from id_mapper.py) ---

def _extract_candidate_ids_from_fasta(fasta_filepath: str) -> Set[str]:
    """
    Extracts potential identifiers from FASTA headers for API mapping.
    """
    candidate_ids = set()
    print(f"Extracting candidate IDs from: {os.path.basename(fasta_filepath)}...")
    try:
        with open(fasta_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc=f"Scanning FASTA headers", leave=False):
                if line.startswith('>'):
                    header = line[1:].strip()
                    # Prioritize db|ID|... format
                    parts = header.split('|')
                    if len(parts) > 1 and parts[1]:
                        candidate_ids.add(parts[1].strip())
                        continue
                    # Fallback to the first word
                    candidate_ids.add(header.split()[0].strip())
    except FileNotFoundError:
        print(f"ERROR: FASTA file not found at {fasta_filepath}")
        return set()
    print(f"Found {len(candidate_ids)} unique candidate IDs for API mapping.")
    return candidate_ids


def _submit_id_mapping_job(ids_to_map: List[str], from_db: str, to_db: str) -> str:
    """Submits a job to the UniProt ID mapping service."""
    payload = {"ids": ",".join(ids_to_map), "from": from_db, "to": to_db}
    response = requests.post("https://rest.uniprot.org/idmapping/run", data=payload)
    response.raise_for_status()
    job_id = response.json().get("jobId")
    if not job_id: raise ValueError("Failed to submit job to UniProt.")
    print(f"  UniProt API job submitted for {len(ids_to_map)} IDs. Job ID: {job_id}")
    return job_id


def _check_job_status(job_id: str) -> str:
    """Checks the status of a submitted job."""
    response = requests.get(f"https://rest.uniprot.org/idmapping/status/{job_id}")
    response.raise_for_status()
    return response.json().get("jobStatus", "UNKNOWN")


def _get_mapping_results(job_id: str) -> List[Dict]:
    """Retrieves the mapping results for a completed job."""
    response = requests.get(f"https://rest.uniprot.org/idmapping/results/{job_id}?format=json")
    response.raise_for_status()
    return response.json().get("results", [])


def run_api_mapping(fasta_path: str, from_database: str, to_database: str, sample_size: Optional[int] = None, random_seed: int = 42) -> Dict[str, str]:
    """
    Main orchestrator for the API mapping workflow.
    Returns a dictionary mapping the original ID to the new canonical ID.
    """
    all_candidate_ids = list(_extract_candidate_ids_from_fasta(fasta_path))
    if not all_candidate_ids: return {}

    if sample_size is not None and len(all_candidate_ids) > sample_size:
        print(f"Randomly sampling {sample_size} IDs from {len(all_candidate_ids)} total candidates.")
        random.seed(random_seed)
        ids_to_process = random.sample(all_candidate_ids, sample_size)
    else:
        ids_to_process = all_candidate_ids

    total_ids = len(ids_to_process)
    processed_mappings: Dict[str, str] = {}
    batch_size = 500  # UniProt API batch size limit
    request_interval = 2  # Seconds between status checks

    print(f"\nStarting UniProt ID mapping for {total_ids} IDs in batches of {batch_size}.")

    for i in range(0, total_ids, batch_size):
        batch_ids = ids_to_process[i:i + batch_size]
        print(f"\nProcessing batch {i // batch_size + 1}/{(total_ids + batch_size - 1) // batch_size}...")
        try:
            job_id = _submit_id_mapping_job(batch_ids, from_database, to_database)
            while True:
                time.sleep(request_interval)
                status = _check_job_status(job_id)
                print(f"  Job {job_id} status: {status}")
                if status == "FINISHED":
                    results = _get_mapping_results(job_id)
                    for entry in results:
                        from_id, to_id = entry.get("from"), entry.get("to", {}).get("primaryAccession")
                        if from_id and to_id: processed_mappings[from_id] = to_id
                    break
                elif status not in ["RUNNING", "QUEUED"]:
                    print(f"  Job {job_id} failed with status: {status}");
                    break
        except Exception as e:
            print(f"  Error processing batch: {e}. Skipping batch.")

    return processed_mappings


# --- Regex-based ID Parsing Functions ---

def extract_canonical_id_and_type(header: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parses a FASTA header and extracts a canonical ID using regex.
    """
    hid = header.strip().lstrip('>')
    up_match = re.match(r"^(?:sp|tr)\|([A-Z0-9]{6,10}(?:-\d+)?)\|", hid, re.IGNORECASE)
    if up_match: return "UniProt", up_match.group(1)
    uniref_match = re.match(r"^(UniRef\d{2,3})_([A-Z0-9]+)", hid, re.IGNORECASE)
    if uniref_match: return "UniProt (from UniRef)", uniref_match.group(2)
    plain_match = re.fullmatch(r"([A-Z0-9]{6,10}(?:-\d+)?)", hid.split()[0])
    if plain_match: return "UniProt (assumed)", plain_match.group(1)
    return "Unknown", hid.split()[0]


def run_regex_mapping(fasta_path: str) -> Dict[str, str]:
    """
    Creates an ID map by parsing all headers in a FASTA file with regex.
    Returns a dictionary mapping original IDs to canonical IDs.
    """
    print(f"Starting Regex ID mapping for: {os.path.basename(fasta_path)}...")
    id_map = {}
    try:
        for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Parsing FASTA with Regex"):
            original_id = record.id
            _, canonical_id = extract_canonical_id_and_type(original_id)
            if canonical_id and canonical_id != original_id:
                id_map[original_id] = canonical_id
    except FileNotFoundError:
        print(f"ERROR: FASTA file not found at {fasta_path}")
    print(f"Regex mapping complete. Found {len(id_map)} potential mappings.")
    return id_map
