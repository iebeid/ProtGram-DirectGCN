import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from Bio import SeqIO
from tqdm.auto import tqdm

# ==============================================================================
# 2. FASTA File Utilities
# ==============================================================================
class FastaUtils:
    """
    A collection of utilities for handling FASTA files, including an
    efficient parser and a memory-safe corpus class for sequence processing.
    """
    # Standard IUPAC amino acid and nucleotide codes for cleaning
    AMINO_ACID_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    NUCLEOTIDE_ALPHABET = "GATCU"

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
    def parse_sequences(fasta_filepaths: List[Union[str, Path]],
                        perform_cleaning: bool = False,
                        min_len: int = 1,
                        max_len: Optional[int] = None,
                        alphabet_type: str = 'protein') -> Iterator[Tuple[str, str]]:
        """
        An efficient FASTA parser that reads one or more FASTA files, yielding
        an ID and sequence for each record. Includes optional cleaning.

        Args:
            fasta_filepaths: List of paths to FASTA files.
            perform_cleaning: If True, applies cleaning filters.
            min_len: Minimum sequence length to keep (if cleaning).
            max_len: Maximum sequence length to keep (if cleaning).
            alphabet_type: 'protein' or 'dna' for validation (if cleaning).
        """
        if perform_cleaning:
            print("  - Cleaning enabled for FASTA parsing.")
            if alphabet_type == 'protein':
                valid_chars = set(FastaUtils.AMINO_ACID_ALPHABET)
            elif alphabet_type == 'dna':
                valid_chars = set(FastaUtils.NUCLEOTIDE_ALPHABET)
            else:
                raise ValueError("alphabet_type must be 'protein' or 'dna'")

        for path_str in tqdm(fasta_filepaths, desc="Parsing FASTA files", leave=False, unit="file"):
            normalized_path = Path(path_str)
            try:
                # --- PERFORMANCE: Avoid reading the file twice. ---
                # We open the file once and iterate through it with SeqIO. The tqdm progress
                # bar will no longer show a total, but this is a major I/O optimization.
                with open(normalized_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for record in tqdm(SeqIO.parse(f, "fasta"), desc=f"  - {normalized_path.name}", leave=False, unit="seq"):
                        protein_id = FastaUtils.extract_id_from_header(record.description)
                        sequence = str(record.seq).upper()

                        if perform_cleaning:
                            if not (min_len <= len(sequence) and (max_len is None or len(sequence) <= max_len)):
                                continue
                            if not set(sequence).issubset(valid_chars):
                                continue

                        yield protein_id, sequence

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