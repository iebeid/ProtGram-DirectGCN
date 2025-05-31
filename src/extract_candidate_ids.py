import re
from tqdm.auto import tqdm
import os
from typing import Set


def extract_candidate_ids_from_fasta(fasta_filepath: str) -> Set[str]:
    """
    Extracts potential identifiers from FASTA headers for later mapping.
    This version tries to get UniProt-like IDs from various common FASTA formats.
    """
    candidate_ids = set()
    print(f"Extracting candidate IDs from: {fasta_filepath} for mapping...")
    current_id_full_header = None

    try:
        with open(fasta_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in tqdm(f, desc=f"Scanning {os.path.basename(fasta_filepath)}"):
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    current_id_full_header = line[1:]

                    # Try to get ID from >db|ID|... format
                    parts = current_id_full_header.split('|')
                    if len(parts) > 1 and parts[1]:
                        candidate_ids.add(parts[1].strip())
                        continue  # Prioritize this if found

                    # Try to get ID from >ID ... format (first word)
                    first_word = current_id_full_header.split()[0]
                    candidate_ids.add(first_word.strip())

                    # Optionally, also try to extract RepID from UniRef headers
                    # Example: >UniRef90_A0A023GPI8 ClusterName RepID=A0A023GPI8_CANLF
                    rep_id_match = re.search(r"RepID=([\w.-]+)", current_id_full_header)
                    if rep_id_match:
                        candidate_ids.add(rep_id_match.group(1).strip())

    except FileNotFoundError:
        print(f"ERROR: FASTA file not found at {fasta_filepath}")
        return set()
    except Exception as e:
        print(f"ERROR: Could not parse FASTA file {fasta_filepath}: {e}")
        return set()

    print(f"Found {len(candidate_ids)} unique candidate IDs.")
    return candidate_ids


if __name__ == "__main__":
    # VITAL: Path to the single FASTA file containing protein sequences to be processed.
    INPUT_FASTA_FILE: str = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/fasta/uniref50/uniref50.fasta"

    # Output file to save these candidate IDs
    CANDIDATE_IDS_FILE = "C:/tmp/Models/candidates/candidate_fasta_ids_for_mapping.txt"

    extracted_ids = extract_candidate_ids_from_fasta(INPUT_FASTA_FILE)

    if extracted_ids:
        with open(CANDIDATE_IDS_FILE, 'w') as f_out:
            for prot_id in extracted_ids:
                f_out.write(f"{prot_id}\n")
        print(f"Saved {len(extracted_ids)} candidate IDs to {CANDIDATE_IDS_FILE}")
    else:
        print("No candidate IDs extracted.")