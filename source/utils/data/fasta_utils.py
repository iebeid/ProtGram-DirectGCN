# ==============================================================================
# MODULE: utils/data/fasta_utils.py
# PURPOSE: Contains utility functions for parsing and handling FASTA files.
# VERSION: 3.0 (Replaced buggy parallel parser with robust Bio.SeqIO.parse)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import re
from pathlib import Path
from typing import Iterator, List, Tuple, Union

from Bio import SeqIO
from tqdm.auto import tqdm

from source.utils.data.data_utils import DataUtils

# ==============================================================================
# 2. FASTA File Utilities
# ==============================================================================

ALPHABET_PROTEIN_NOCUT = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
ALPHABET_PROTEIN_CUT = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET_DNA = "GATCRYWSMKHBVDN"
ALPHABET_RNA = "GAUCRYWSMKHBVDN"


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
    def clean_sequence(sequence: str, alphabet_type: str = 'protein') -> str:
        """
        Removes ambiguous characters from a sequence string.
        """
        if alphabet_type == 'protein':
            return re.sub(f"[^{FastaUtils.AMINO_ACID_ALPHABET}]", "", sequence)
        elif alphabet_type == 'dna':
            return re.sub(f"[^{FastaUtils.NUCLEOTIDE_ALPHABET}]", "", sequence)
        else:
            return sequence

    @staticmethod
    def parse_sequences(
            fasta_filepaths: List[Union[str, Path]],
            perform_cleaning: bool = False,
            min_len: int = 1,
            max_len: int = 100000,
            alphabet_type: str = 'protein'
    ) -> Iterator[Tuple[str, str]]:
        """
        Parses one or more FASTA files and yields sequences. This implementation
        uses the robust and memory-efficient Bio.SeqIO.parse iterator, which is
        the standard and safest way to handle FASTA files.
        """
        # --- DEFINITIVE FIX for Correctness: Replace buggy parallel parser ---
        # The previous line-chunking parallel parser was not safe for the FASTA
        # format and could lead to data loss or corruption. Using the standard
        # Bio.SeqIO.parse iterator guarantees correctness and is still highly
        # memory-efficient.
        if perform_cleaning:
            print("  - Cleaning enabled for FASTA parsing.")

        for file_path in fasta_filepaths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for record in SeqIO.parse(f, "fasta"):
                        header = FastaUtils.extract_id_from_header(record.description)
                        sequence = str(record.seq)
                        if perform_cleaning:
                            sequence = FastaUtils.clean_sequence(sequence, alphabet_type)
                        if min_len <= len(sequence) <= max_len:
                            yield header, sequence
            except FileNotFoundError:
                print(f"Warning: FASTA file not found: {file_path}")
                continue

    @staticmethod
    def check_uniprot_header_compatibility(fasta_paths: List[Path], sample_size: int) -> float:
        """
        Samples FASTA files to determine the compatibility score with the fast regex parser.
        Returns the score (fraction of headers that look like UniProt headers).
        """
        if not fasta_paths:
            return 0.0

        def header_iterator(paths: List[Path]) -> Iterator[str]:
            for path in paths:
                if not path.exists(): continue
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.startswith('>'):
                            yield line.strip()

        all_headers_iterator = header_iterator(fasta_paths)
        sampled_headers = DataUtils.reservoir_sample(all_headers_iterator, sample_size)

        if not sampled_headers:
            return 0.0

        uniprot_like_ids = 0
        for header in sampled_headers:
            if FastaUtils.extract_id_from_header(header) != header.lstrip('>').split()[0]:
                uniprot_like_ids += 1

        return uniprot_like_ids / len(sampled_headers)

    class FastaCorpus:
        """A memory-efficient corpus for Word2Vec that reads from FASTA files."""

        def __init__(self, fasta_files: List[Union[str, Path]]):
            self.fasta_files = [Path(f) for f in fasta_files]

        def __iter__(self) -> Iterator[List[str]]:
            # --- REFACTOR: Call parse_sequences once with all files for efficiency ---
            # This now correctly calls the parallelized parser.
            for _, sequence in FastaUtils.parse_sequences(
                self.fasta_files,
                perform_cleaning=True,
                min_len=1,
                max_len=100000,
                alphabet_type='protein'
            ):
                if sequence:
                    yield list(sequence)