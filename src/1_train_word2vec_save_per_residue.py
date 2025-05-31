import os
import time
import pickle
import gc
import glob
from collections import defaultdict
from typing import List, Iterator, Tuple, Dict  # Removed unused 'Dict' from here for a moment

import numpy as np
import h5py
from gensim.models import Word2Vec
from tqdm.auto import tqdm

# --- User Configuration ---
# Describe the dataset/run for output filenames
DATASET_TAG = "uniref50_w2v_per_residue_memfix"  # Updated tag

# Input FASTA files:
FASTA_INPUT_DIR = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/fasta/uniref50/"
FASTA_FILE_PATHS_LIST = None

# Output paths
OUTPUT_BASE_DIR = "C:/tmp/Models/word2vec_per_residue/"  # Using local path that worked previously for outputs
WORD2VEC_MODEL_FILENAME = f"word2vec_char_model_{DATASET_TAG}.model"
PER_RESIDUE_H5_FILENAME = f"word2vec_per_residue_embeddings_{DATASET_TAG}.h5"

# Word2Vec Hyperparameters
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 1
W2V_WORKERS = max(1, os.cpu_count() - 2 if os.cpu_count() else 1)
W2V_EPOCHS = 1  # Consider increasing for better embeddings if time permits
W2V_SG = 1
W2V_COMPUTE_LOSS = False


# --- End User Configuration ---

def count_sequences_in_fasta_fast(filepath: str) -> int:
    filepath = os.path.normpath(filepath)
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
        return 0  # Return 0 if counting fails, so sum still works
    return count


def simple_fasta_parser(fasta_filepath: str) -> Iterator[Tuple[str, str]]:
    fasta_filepath = os.path.normpath(fasta_filepath)
    current_id_full_header = None
    uniprot_id = None
    sequence_parts = []
    try:
        with open(fasta_filepath, 'r', encoding='utf-8', errors='ignore') as f:  # Added errors='ignore' for robustness
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if current_id_full_header and sequence_parts:
                        final_id = uniprot_id if uniprot_id else current_id_full_header.split()[0]
                        yield final_id, "".join(sequence_parts)
                    current_id_full_header = line[1:]
                    # Attempt to extract a cleaner ID (UniProt ID or first word of header)
                    parts = current_id_full_header.split('|')
                    if len(parts) > 1 and parts[1]:
                        uniprot_id = parts[1]
                    else:
                        uniprot_id = current_id_full_header.split()[0]
                    sequence_parts = []
                elif current_id_full_header:
                    sequence_parts.append(line.upper())
            if current_id_full_header and sequence_parts:  # Yield last sequence
                final_id = uniprot_id if uniprot_id else current_id_full_header.split()[0]
                yield final_id, "".join(sequence_parts)
    except FileNotFoundError:
        print(f"Warning: FASTA file not found: {fasta_filepath}")
    except Exception as e:
        print(f"Warning: Error parsing FASTA file {fasta_filepath}: {e}")


class FastaCorpusForWord2Vec:
    def __init__(self, fasta_file_paths_list: List[str], total_sequences_for_tqdm: int = 0):  # Default total to 0
        self.fasta_file_paths_list = fasta_file_paths_list
        self.total_sequences_for_tqdm = total_sequences_for_tqdm
        self.pbar = None

    def __iter__(self) -> Iterator[List[str]]:
        if self.pbar is not None:
            self.pbar.reset(total=self.total_sequences_for_tqdm if self.total_sequences_for_tqdm > 0 else None)
        elif self.total_sequences_for_tqdm is not None and self.total_sequences_for_tqdm > 0:
            self.pbar = tqdm(total=self.total_sequences_for_tqdm, desc="Word2Vec Corpus Iteration")

        for f_path in self.fasta_file_paths_list:
            for _, sequence_str in simple_fasta_parser(f_path):
                if sequence_str:
                    if self.pbar: self.pbar.update(1)
                    yield list(sequence_str)
        if self.pbar: self.pbar.close(); self.pbar = None


def main():
    print("--- Script A: Word2Vec Model Training & Per-Residue Embedding Generation ---")
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    word2vec_model_path = os.path.normpath(os.path.join(OUTPUT_BASE_DIR, WORD2VEC_MODEL_FILENAME))
    per_residue_h5_path = os.path.normpath(os.path.join(OUTPUT_BASE_DIR, PER_RESIDUE_H5_FILENAME))

    actual_fasta_files = []
    if FASTA_FILE_PATHS_LIST:
        actual_fasta_files = [os.path.normpath(f) for f in FASTA_FILE_PATHS_LIST if os.path.isfile(os.path.normpath(f))]
    elif FASTA_INPUT_DIR and os.path.isdir(os.path.normpath(FASTA_INPUT_DIR)):
        norm_fasta_input_dir = os.path.normpath(FASTA_INPUT_DIR)
        print(f"Scanning directory '{norm_fasta_input_dir}' for FASTA files...")
        for ext in ('*.fasta', '*.fas', '*.fa', '*.fna'):
            actual_fasta_files.extend(glob.glob(os.path.join(norm_fasta_input_dir, ext)))
    else:
        print(f"Error: FASTA input directory not found or invalid: {FASTA_INPUT_DIR}")
        return

    if not actual_fasta_files: print("Error: No FASTA files found. Exiting."); return
    print(f"Found {len(actual_fasta_files)} FASTA file(s) to process.")

    print("Performing initial scan to count total sequences for progress bar...")
    total_sequences = sum(count_sequences_in_fasta_fast(f) for f in actual_fasta_files)
    if total_sequences == 0: print("Warning: No sequences counted. Word2Vec training may be empty or fail.")

    print(f"Total sequences for Word2Vec training: {total_sequences}")
    corpus_iterator = FastaCorpusForWord2Vec(actual_fasta_files, total_sequences_for_tqdm=total_sequences)

    print(
        f"Training Word2Vec model (Size:{W2V_VECTOR_SIZE}, Window:{W2V_WINDOW}, MinCount:{W2V_MIN_COUNT}, Epochs:{W2V_EPOCHS}, Workers:{W2V_WORKERS})...")
    start_time = time.time()
    word2vec_model = Word2Vec(sentences=corpus_iterator, vector_size=W2V_VECTOR_SIZE, window=W2V_WINDOW,
                              min_count=W2V_MIN_COUNT, workers=W2V_WORKERS, sg=W2V_SG, epochs=W2V_EPOCHS,
                              compute_loss=W2V_COMPUTE_LOSS)
    print(f"Word2Vec model trained in {time.time() - start_time:.2f} seconds.")
    print(f"Saving Word2Vec model to: {word2vec_model_path}")
    word2vec_model.save(word2vec_model_path);
    print("Word2Vec model saved.")

    print(f"\nGenerating and saving per-residue Word2Vec embeddings to: {per_residue_h5_path}")
    with h5py.File(per_residue_h5_path, 'w') as hf:
        sequences_embedded_count = 0
        # Use a new tqdm progress bar for this part
        total_files_to_process_for_embedding = len(actual_fasta_files)
        # If total_sequences is 0, use number of files for progress bar, otherwise use total sequences for a rough guide
        pbar_embedding = tqdm(total=total_sequences if total_sequences > 0 else total_files_to_process_for_embedding,
                              desc="Generating/Saving Per-Residue Embeddings")

        processed_in_current_file_count = 0

        for fasta_file_idx, fasta_file in enumerate(actual_fasta_files):
            # print(f"Processing file {fasta_file_idx+1}/{total_files_to_process_for_embedding}: {fasta_file}")
            for prot_id, sequence in simple_fasta_parser(fasta_file):
                if not sequence:
                    # print(f"Warning: Skipping protein {prot_id} from {fasta_file} due to empty sequence.")
                    if total_sequences == 0: pbar_embedding.update(1)  # Update per file if no seq count
                    continue

                # Sanitize protein ID for HDF5 dataset name (HDF5 doesn't like slashes)
                safe_prot_id = str(prot_id).replace('/', '_SLASH_').replace('\\', '_BSLASH_')

                if safe_prot_id in hf:
                    # print(f"Warning: Dataset for {safe_prot_id} (original: {prot_id}) already exists. Skipping.")
                    if total_sequences > 0: pbar_embedding.update(1)  # Update per sequence
                    continue

                try:
                    # Create dataset first, then fill it iteratively
                    dset = hf.create_dataset(safe_prot_id, shape=(len(sequence), W2V_VECTOR_SIZE), dtype=np.float32)
                    for i, char_residue in enumerate(sequence):
                        if char_residue in word2vec_model.wv:
                            dset[i] = word2vec_model.wv[char_residue]
                        else:
                            dset[i] = np.zeros(W2V_VECTOR_SIZE, dtype=np.float32)
                    sequences_embedded_count += 1
                except Exception as e:
                    print(
                        f"Error processing or saving embeddings for protein {prot_id} (safe_id: {safe_prot_id}) from {fasta_file}: {e}")

                if total_sequences > 0: pbar_embedding.update(1)  # Update per sequence

            if total_sequences == 0:  # If no sequence count, update pbar per file processed
                pbar_embedding.update(1)

        pbar_embedding.close()
        hf.attrs['embedding_type'] = 'word2vec_per_residue'
        hf.attrs['vector_size'] = W2V_VECTOR_SIZE
        hf.attrs['dataset_tag'] = DATASET_TAG
        print(f"Saved per-residue embeddings for {sequences_embedded_count} proteins.")

    print("Script A finished.")


if __name__ == "__main__":
    main()