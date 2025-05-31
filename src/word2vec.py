import os
import time
import pickle
import gc
import glob
from collections import defaultdict
from typing import List, Iterator, Tuple, Dict

import numpy as np
import h5py
from gensim.models import Word2Vec
from tqdm.auto import tqdm

# --- User Configuration ---
# Describe the dataset/run for output filenames
DATASET_TAG = "uniref50_w2v_per_residue"  # EXAMPLE: CHANGE THIS!

# Input FASTA files:
# Option 1: Provide a directory path. The script will scan for *.fasta, *.fas, *.fa, *.fna files.
FASTA_INPUT_DIR = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/fasta/uniref50/" 
# Option 2: Provide a list of full paths to FASTA files. Overrides directory if not None.
# FASTA_FILE_PATHS_LIST = ["/path/to/your/file1.fasta", "/path/to/your/file2.fasta"]
FASTA_FILE_PATHS_LIST = None

# Output paths
OUTPUT_BASE_DIR = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/word2vec_outputs"
WORD2VEC_MODEL_FILENAME = f"word2vec_char_model_{DATASET_TAG}.model"
PER_RESIDUE_H5_FILENAME = f"word2vec_per_residue_embeddings_{DATASET_TAG}.h5"

# Word2Vec Hyperparameters
W2V_VECTOR_SIZE = 100  # Native embedding size from Word2Vec
W2V_WINDOW = 5
W2V_MIN_COUNT = 5      # Increased min_count for speed and better generalizability
W2V_WORKERS = max(1, os.cpu_count() - 2 if os.cpu_count() else 1) # Leave some cores free
W2V_EPOCHS = 5         # Gensim's default, 1 was too low
W2V_SG = 1             # 1 for skip-gram, 0 for CBOW
W2V_COMPUTE_LOSS = False # Set to True to monitor, False for slight speed up

# --- End User Configuration ---

# --- Helper Functions (from your models.py) ---
def count_sequences_in_fasta_fast(filepath):
    count = 0
    try:
        with open(filepath, 'rb') as f:
            chunk_size = 32 * 1024 * 1024 
            while True:
                chunk = f.read(chunk_size)
                if not chunk: break
                count += chunk.count(b'>')
    except Exception as e:
        print(f"Warning: Fast count failed for {filepath}: {e}")
        return 0 
    return count

def simple_fasta_parser(fasta_filepath: str) -> Iterator[Tuple[str, str]]:
    current_id_full_header = None
    uniprot_id = None
    sequence_parts = []
    try:
        with open(fasta_filepath, 'r', encoding='utf-8') as f: # Specify encoding
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if current_id_full_header and sequence_parts:
                        yield uniprot_id if uniprot_id else current_id_full_header, "".join(sequence_parts)
                    current_id_full_header = line[1:]
                    parts = current_id_full_header.split('|')
                    if len(parts) > 1 and parts[1]: # Standard UniProt ID like >db|UniProtKB_ID|EntryName
                        uniprot_id = parts[1]
                    else: # Fallback to first part of header
                        uniprot_id = current_id_full_header.split()[0]
                    sequence_parts = []
                elif current_id_full_header: # Ensure we are inside a sequence entry
                    sequence_parts.append(line.upper()) # Ensure uppercase for consistency
            if current_id_full_header and sequence_parts: # Yield last sequence
                yield uniprot_id if uniprot_id else current_id_full_header, "".join(sequence_parts)
    except FileNotFoundError:
        print(f"Warning: FASTA file not found: {fasta_filepath}")
    except Exception as e:
        print(f"Warning: Error parsing FASTA file {fasta_filepath}: {e}")

class FastaCorpusForWord2Vec:
    def __init__(self, fasta_file_paths_list: List[str], total_sequences_for_tqdm: int = None):
        self.fasta_file_paths_list = fasta_file_paths_list
        self.total_sequences_for_tqdm = total_sequences_for_tqdm
        self.pbar = None

    def __iter__(self) -> Iterator[List[str]]:
        # Reset progress bar for each epoch if Word2Vec iterates multiple times
        if self.pbar is not None:
            self.pbar.reset(total=self.total_sequences_for_tqdm)
        elif self.total_sequences_for_tqdm is not None and self.total_sequences_for_tqdm > 0:
            self.pbar = tqdm(total=self.total_sequences_for_tqdm, desc="Word2Vec Corpus Iteration")

        for f_path in self.fasta_file_paths_list:
            for _, sequence_str in simple_fasta_parser(f_path):
                if sequence_str: # Ensure sequence is not empty
                    if self.pbar: self.pbar.update(1)
                    yield list(sequence_str) # Yield list of characters (residues)
        if self.pbar:
            self.pbar.close()
            self.pbar = None # Important to allow re-creation for next epoch pass by Word2Vec

def main():
    print("--- Script A: Word2Vec Model Training & Per-Residue Embedding Generation ---")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    word2vec_model_path = os.path.join(OUTPUT_BASE_DIR, WORD2VEC_MODEL_FILENAME)
    per_residue_h5_path = os.path.join(OUTPUT_BASE_DIR, PER_RESIDUE_H5_FILENAME)

    actual_fasta_files = []
    if FASTA_FILE_PATHS_LIST:
        actual_fasta_files = [f for f in FASTA_FILE_PATHS_LIST if os.path.isfile(f)]
    elif FASTA_INPUT_DIR and os.path.isdir(FASTA_INPUT_DIR):
        print(f"Scanning directory '{FASTA_INPUT_DIR}' for FASTA files...")
        for ext in ('*.fasta', '*.fas', '*.fa', '*.fna'):
            actual_fasta_files.extend(glob.glob(os.path.join(FASTA_INPUT_DIR, ext)))
    
    if not actual_fasta_files:
        print("Error: No FASTA files found. Exiting.")
        return
    print(f"Found {len(actual_fasta_files)} FASTA file(s) to process.")

    print("Performing initial scan to count total sequences for progress bar...")
    total_sequences = sum(count_sequences_in_fasta_fast(f) for f in actual_fasta_files)
    if total_sequences == 0:
        print("Warning: No sequences counted in FASTA files. Word2Vec training might be empty or fail.")
        # Still proceed, FastaCorpusForWord2Vec will handle empty iteration

    print(f"Total sequences for Word2Vec training: {total_sequences}")
    
    corpus_iterator = FastaCorpusForWord2Vec(actual_fasta_files, total_sequences_for_tqdm=total_sequences)

    print(f"Training Word2Vec model (Size:{W2V_VECTOR_SIZE}, Window:{W2V_WINDOW}, MinCount:{W2V_MIN_COUNT}, Epochs:{W2V_EPOCHS}, Workers:{W2V_WORKERS})...")
    start_time = time.time()
    word2vec_model = Word2Vec(
        sentences=corpus_iterator,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=W2V_WORKERS,
        sg=W2V_SG,
        epochs=W2V_EPOCHS,
        compute_loss=W2V_COMPUTE_LOSS 
    )
    print(f"Word2Vec model trained in {time.time() - start_time:.2f} seconds.")
    
    print(f"Saving Word2Vec model to: {word2vec_model_path}")
    word2vec_model.save(word2vec_model_path)
    print("Word2Vec model saved.")

    print(f"\nGenerating and saving per-residue Word2Vec embeddings to: {per_residue_h5_path}")
    with h5py.File(per_residue_h5_path, 'w') as hf:
        sequences_embedded_count = 0
        for fasta_file in tqdm(actual_fasta_files, desc="Generating Per-Residue Embeddings"):
            for prot_id, sequence in simple_fasta_parser(fasta_file):
                if not sequence: continue
                residue_vectors = []
                for char_residue in sequence:
                    if char_residue in word2vec_model.wv:
                        residue_vectors.append(word2vec_model.wv[char_residue])
                    else:
                        # Handle out-of-vocabulary characters (e.g., use a zero vector or skip)
                        residue_vectors.append(np.zeros(W2V_VECTOR_SIZE, dtype=np.float32)) 
                
                if residue_vectors:
                    hf.create_dataset(prot_id, data=np.array(residue_vectors, dtype=np.float32))
                    sequences_embedded_count +=1
        hf.attrs['embedding_type'] = 'word2vec_per_residue'
        hf.attrs['vector_size'] = W2V_VECTOR_SIZE
        hf.attrs['dataset_tag'] = DATASET_TAG
        print(f"Saved per-residue embeddings for {sequences_embedded_count} proteins.")

    print("Script A finished.")

if __name__ == "__main__":
    main()