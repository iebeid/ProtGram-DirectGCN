import requests
import time
import os
import random  # Added for sampling
from typing import List, Dict, Set

# --- Configuration ---
CANDIDATE_IDS_FILE = "C:/tmp/Models/candidates/candidate_fasta_ids_for_mapping.txt"  # Input file with one ID per line
OUTPUT_MAPPING_FILE = "C:/tmp/Models/candidates/uniprot_api_id_map.tsv"  # Output file for the mappings
UNIPROT_ID_MAPPING_URL = "https://rest.uniprot.org/idmapping"

NUM_IDS_TO_SAMPLE = 10000  # Number of IDs to randomly sample from the input file
SEED_FOR_SAMPLING = 42  # Seed for reproducible random sampling

# API Parameters (Adjust as needed based on UniProt guidelines)
BATCH_SIZE = 500  # Number of IDs to submit in each API request (even for the sample)
REQUEST_INTERVAL = 2  # Seconds to wait between batch requests

FROM_DATABASE = "UniRef50"  # Example: 'UniProtKB_AC-ID', 'UniRef100', 'GeneID', 'EMBL'
# Adjust based on the primary type of your candidate IDs.
TO_DATABASE = "UniProtKB"  # Typically, map to reviewed UniProtKB primary accessions.


# --- End Configuration ---

def read_ids_from_file(filepath: str) -> List[str]:
    """Reads IDs from a file, one ID per line."""
    if not os.path.exists(filepath):
        print(f"Error: Candidate ID file not found at '{filepath}'")
        return []
    with open(filepath, 'r') as f:
        ids = [line.strip() for line in f if line.strip()]
    print(f"Read {len(ids)} total candidate IDs from '{filepath}'")
    return ids


def submit_id_mapping_job(ids_to_map: List[str], from_db: str, to_db: str) -> str:
    """Submits a job to the UniProt ID mapping service and returns the job ID."""
    payload = {
        "ids": ",".join(ids_to_map),
        "from": from_db,
        "to": to_db,
    }
    response = requests.post(f"{UNIPROT_ID_MAPPING_URL}/run", data=payload)
    response.raise_for_status()
    job_id = response.json().get("jobId")
    if not job_id:
        raise ValueError("Failed to submit job or retrieve job ID from UniProt.")
    print(f"  Submitted job for {len(ids_to_map)} IDs. Job ID: {job_id}")
    return job_id


def check_job_status(job_id: str) -> Dict:
    """Checks the status of a submitted job."""
    status_url = f"{UNIPROT_ID_MAPPING_URL}/status/{job_id}"
    response = requests.get(status_url)
    response.raise_for_status()
    return response.json()


def get_mapping_results(job_id: str) -> List[Dict]:
    """Retrieves the mapping results for a completed job."""
    results_url = f"{UNIPROT_ID_MAPPING_URL}/results/{job_id}?format=json"
    response = requests.get(results_url)
    response.raise_for_status()
    results_data = response.json().get("results", [])
    return results_data


def main():
    # Set seed for reproducible sampling
    random.seed(SEED_FOR_SAMPLING)

    all_candidate_ids_from_file: List[str] = read_ids_from_file(CANDIDATE_IDS_FILE)
    if not all_candidate_ids_from_file:
        return

    ids_to_process: List[str]

    if len(all_candidate_ids_from_file) > NUM_IDS_TO_SAMPLE:
        print(
            f"Original candidate ID count: {len(all_candidate_ids_from_file)}. Randomly sampling {NUM_IDS_TO_SAMPLE} IDs.")
        ids_to_process = random.sample(all_candidate_ids_from_file, NUM_IDS_TO_SAMPLE)
        with open("C:/tmp/Models/candidates/idmapping_2025_05_30.tsv", "w") as file:
            for item in ids_to_process:
                file.write(str(item) + "\n")
    else:
        print(
            f"Original candidate ID count: {len(all_candidate_ids_from_file)}. Processing all available IDs (less than or equal to {NUM_IDS_TO_SAMPLE}).")
        ids_to_process = all_candidate_ids_from_file

    if not ids_to_process:
        print("No IDs selected for processing (list is empty after sampling attempt or originally).")
        return

    total_ids_to_submit = len(ids_to_process)
    processed_mappings: Dict[str, str] = {}
    failed_ids_in_sample: Set[str] = set()  # Tracks IDs from the sample that failed API processing specifically

    print(f"\nStarting UniProt ID mapping for the {total_ids_to_submit} selected (sampled) IDs.")
    print(f"These IDs will be submitted in batches of up to {BATCH_SIZE} to the UniProt API.")

    for i in range(0, total_ids_to_submit, BATCH_SIZE):
        batch_ids = ids_to_process[i:i + BATCH_SIZE]
        print(
            f"\nProcessing batch {i // BATCH_SIZE + 1}/{(total_ids_to_submit + BATCH_SIZE - 1) // BATCH_SIZE} (IDs {i + 1}-{min(i + BATCH_SIZE, total_ids_to_submit)} of the selected sample)...")

        try:
            job_id = submit_id_mapping_job(batch_ids, FROM_DATABASE, TO_DATABASE)

            while True:
                time.sleep(REQUEST_INTERVAL * 2)
                status_response = check_job_status(job_id)
                job_status = status_response.get("jobStatus")
                print(f"  Job {job_id} status: {job_status}")

                if job_status == "FINISHED":
                    results = get_mapping_results(job_id)
                    if results:
                        for mapping_entry in results:
                            from_id = mapping_entry.get("from")
                            to_data = mapping_entry.get("to")
                            if from_id and to_data:
                                if isinstance(to_data, dict) and "primaryAccession" in to_data:
                                    to_id = to_data["primaryAccession"]
                                    processed_mappings[from_id] = to_id
                                elif isinstance(to_data, str):
                                    to_id = to_data
                                    processed_mappings[from_id] = to_id
                                else:
                                    print(
                                        f"    Warning: No 'primaryAccession' or suitable string in 'to' field for {from_id}: {to_data}")
                                    if from_id in batch_ids: failed_ids_in_sample.add(from_id)
                            else:
                                print(f"    Warning: Incomplete mapping entry: {mapping_entry}")
                                if from_id and from_id in batch_ids: failed_ids_in_sample.add(from_id)
                    else:
                        print(f"  Warning: Job {job_id} finished but returned no results for the batch.")
                        for bid in batch_ids: failed_ids_in_sample.add(bid)
                    break
                elif job_status in ["RUNNING", "QUEUED"]:
                    pass
                else:
                    print(
                        f"  Error: Job {job_id} failed or has unexpected status: {job_status}. Response: {status_response}")
                    for bid in batch_ids: failed_ids_in_sample.add(bid)
                    break

            time.sleep(REQUEST_INTERVAL)

        except requests.exceptions.RequestException as e:
            print(
                f"  Error during API request for batch starting at ID {ids_to_process[i] if i < len(ids_to_process) else 'N/A'}: {e}")
            for bid in batch_ids: failed_ids_in_sample.add(bid)
            print(f"  Skipping this batch due to error.")
            time.sleep(REQUEST_INTERVAL * 5)
        except ValueError as e:
            print(f"  Error processing batch: {e}")
            for bid in batch_ids: failed_ids_in_sample.add(bid)

    print("\n--- Mapping Summary ---")
    print(f"Attempted to map {total_ids_to_submit} randomly sampled IDs.")
    print(f"Successfully obtained mappings for {len(processed_mappings)} of these IDs.")

    unmapped_count = 0
    with open(OUTPUT_MAPPING_FILE, 'w') as f_out:
        f_out.write(f"From_ID\tTo_UniProt_Accession\n")
        for original_id in ids_to_process:  # Iterate over the sampled list
            if original_id in processed_mappings:
                f_out.write(f"{original_id}\t{processed_mappings[original_id]}\n")
            else:
                unmapped_count += 1
                f_out.write(f"{original_id}\tNOT_MAPPED\n")

    print(f"Wrote results for {total_ids_to_submit} sampled IDs to '{OUTPUT_MAPPING_FILE}'.")
    print(f"  - {len(processed_mappings)} IDs were successfully mapped.")
    print(
        f"  - {unmapped_count} IDs were 'NOT_MAPPED' (either no mapping found by UniProt or an error occurred for that ID/batch).")
    if failed_ids_in_sample:
        print(
            f"  Specifically, {len(failed_ids_in_sample)} IDs from the sample encountered direct processing errors or had problematic mapping results.")


if __name__ == "__main__":
    main()