import os
import h5py
import time
import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
import gc
from typing import List, Optional, Set, Dict, Iterator, Tuple
import glob
import concurrent.futures
import random
import pandas as pd  # For loading the mapping file
import re  # For potentially more refined initial ID extraction

# --- User Configuration ---
SEED = 42  #

INPUT_FASTA_FILE: str = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/fasta/uniref50/uniref50.fasta"  #
INPUT_PER_RESIDUE_H5_DIR: str = "C:/tmp/Models/output_char_embeddings_GlobalCharGraph_RandInitFeat_v2/32/"  #
OUTPUT_BASE_DIR = "C:/tmp/Models/protein_embeddings_per_input_h5_v1/"  #
OUTPUT_FILENAME_PREFIX = "pooled_proteins_from"  #
RUN_TAG = "CharEmb_ASCII_FIX"  #

# VITAL: Path to your pre-computed UniProt ID mapping file (TSV/CSV)
# This file should have at least two columns: one for the IDs extracted from FASTA headers (e.g., 'From_ID')
# and one for the canonical UniProt Accessions they map to (e.g., 'To_UniProt_Accession').
ID_MAPPING_FILE_PATH: str = "C:/tmp/Models/candidates/idmapping_2025_05_30.tsv"  # <--- UPDATE THIS PATH
# Specify the column names from your mapping file
MAPPING_FILE_FROM_ID_COLUMN: str = "From"  # Or "your_id", "Query", etc.
MAPPING_FILE_TO_ID_COLUMN: str = "To"  # Or "Entry", "MappedKB_AC", etc.

POOLING_STRATEGY = 'mean'  #
APPLY_PCA = True  #
COMMON_EMBEDDING_DIM_PCA = 64  #
NUM_PROCESSING_WORKERS = max(1,
                             os.cpu_count() - 8 if os.cpu_count() and os.cpu_count() > 2 else 1)  # Adjusted default #
SAMPLE_N_SEQUENCES: Optional[int] = 5000  # Set to a number to sample, or None to process all #
# --- End User Configuration ---

np.random.seed(SEED)  #
if SAMPLE_N_SEQUENCES is not None and SAMPLE_N_SEQUENCES > 0:  #
    random.seed(SEED)  #


def load_id_mapping(mapping_filepath: str, from_col: str, to_col: str) -> Dict[str, str]:
    """Loads the pre-computed ID mapping into a dictionary."""
    if not os.path.exists(mapping_filepath):
        print(f"WARNING: ID Mapping file not found at {mapping_filepath}. Proceeding without ID mapping.")
        return {}
    try:
        # Adjust sep='\t' if your file is comma-separated, etc.
        map_df = pd.read_csv(mapping_filepath, sep='\t', dtype=str)
        if from_col not in map_df.columns or to_col not in map_df.columns:
            print(f"ERROR: Mapping file {mapping_filepath} must contain columns '{from_col}' and '{to_col}'.")
            print(f"Found columns: {map_df.columns.tolist()}")
            return {}

        # Handle cases where one 'From_ID' might map to multiple 'To_ID's (e.g. if mapping produced multiple results)
        # For simplicity, we take the first one if multiple exist, or ensure it's unique.
        # UniProt mapping tool often provides a single best mapping.
        mapping_dict = pd.Series(map_df[to_col].values, index=map_df[from_col]).to_dict()

        # If there can be multiple 'To' for a 'From', and you want to ensure one-to-one,
        # you might need to group by 'from_col' and take the first 'to_col'.
        # Example: mapping_dict = map_df.groupby(from_col)[to_col].first().to_dict()

        print(f"Loaded {len(mapping_dict)} ID mappings from {mapping_filepath}")
        return mapping_dict
    except Exception as e:
        print(f"Error loading ID mapping file {mapping_filepath}: {e}")
        return {}


def parse_fasta_sequences_with_mapping(
        fasta_filepath: str,
        id_mapping_dict: Dict[str, str]
) -> Iterator[Tuple[str, str]]:
    fasta_filepath = os.path.normpath(fasta_filepath)  #
    current_id_full_header = None  #
    final_uniprot_id_to_yield = None
    sequence_parts = []  #
    print(f"Parsing FASTA sequences from: {fasta_filepath} (using pre-computed ID map)")  #

    sequences_without_mapping = 0
    sequences_with_mapping = 0

    try:  #
        # ... (tqdm setup as before) ...
        total_lines = None
        try:
            if os.path.getsize(fasta_filepath) < 500 * 1024 * 1024:
                with open(fasta_filepath, 'r', encoding='utf-8', errors='ignore') as f_count:
                    total_lines = sum(1 for _ in f_count)
        except Exception:
            pass

        with open(fasta_filepath, 'r', encoding='utf-8', errors='ignore') as f:  #
            file_iterator = tqdm(f, desc=f"Reading {os.path.basename(fasta_filepath)}", total=total_lines, unit="lines",
                                 #
                                 leave=False, unit_scale=True)  #
            for line in file_iterator:  #
                line = line.strip()  #
                if not line:  #
                    continue  #

                if line.startswith('>'):  #
                    if current_id_full_header and sequence_parts:  #
                        if final_uniprot_id_to_yield:  # Only yield if we have a valid ID
                            yield final_uniprot_id_to_yield, "".join(sequence_parts)  #
                        # Reset for next entry
                        final_uniprot_id_to_yield = None

                    current_id_full_header = line[1:]  #
                    sequence_parts = []  #

                    # Step 1: Extract a "raw" candidate ID from the header.
                    # This logic should ideally match how IDs were extracted for Step 1 (Candidate ID List Generation).
                    raw_id_candidate = None
                    parts = current_id_full_header.split('|')  #
                    if len(parts) > 1 and parts[1]:  #
                        raw_id_candidate = parts[1].strip()
                    else:  #
                        raw_id_candidate = current_id_full_header.split()[0].strip()  #

                    # Attempt to also get RepID as another candidate if the first one fails
                    # This is a fallback if the primary candidate isn't in the map.
                    # Note: The id_mapping_dict should ideally contain mappings for *all* ID types
                    # you submitted to UniProt for mapping.
                    rep_id_candidate = None
                    rep_id_match = re.search(r"RepID=([\w.-]+)", current_id_full_header)
                    if rep_id_match:
                        rep_id_candidate = rep_id_match.group(1).strip()

                    # Step 2: Look up in the pre-computed map
                    # Prioritize looking up the most likely candidate first
                    mapped_id = id_mapping_dict.get(raw_id_candidate)

                    if not mapped_id and rep_id_candidate:  # If first candidate failed, try RepID
                        mapped_id = id_mapping_dict.get(rep_id_candidate)

                    if mapped_id:
                        final_uniprot_id_to_yield = mapped_id
                        sequences_with_mapping += 1
                    else:
                        # Fallback: If no mapping found for any candidate from this header.
                        # Decide whether to skip, use a raw ID, or raise an error.
                        # For robustness, we might skip or use a placeholder that won't match downstream if strictness is required.
                        # Or, use the best raw candidate and let downstream handle mismatches.
                        # print(f"Warning: No mapping found for header candidates from '{current_id_full_header[:50]}...'. Raw primary candidate: '{raw_id_candidate}'. Skipping this sequence.")
                        final_uniprot_id_to_yield = None  # Explicitly set to None to skip yielding if no map
                        sequences_without_mapping += 1

                elif current_id_full_header:  # check ensures we are processing sequence lines #
                    sequence_parts.append(line.upper())  #

            # Yield the last sequence in the file #
            if current_id_full_header and sequence_parts:  #
                if final_uniprot_id_to_yield:  # Only yield if we have a valid ID for the last entry
                    yield final_uniprot_id_to_yield, "".join(sequence_parts)  #

        print(f"FASTA parsing: {sequences_with_mapping} sequences mapped to UniProt ID.")
        if sequences_without_mapping > 0:
            print(
                f"FASTA parsing WARNING: {sequences_without_mapping} sequences could not be mapped to a UniProt ID using the provided map and were skipped.")

    except FileNotFoundError:  #
        print(f"FATAL ERROR: FASTA file not found at {fasta_filepath}. Please check INPUT_FASTA_FILE path.")  #
        raise  #
    except Exception as e:  #
        print(f"FATAL ERROR: Could not parse FASTA file {fasta_filepath}: {e}")  #
        raise  #


# load_single_h5_vocabulary and pool_sequence_embeddings_worker remain the same as in your provided script
def load_single_h5_vocabulary(h5_file_path: str) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[int]]:  #
    """Loads residue embeddings from a single HDF5 file."""  #
    residue_vocab = {}  #
    embedding_dim = None  #
    h5_path = os.path.normpath(h5_file_path)  #
    if not os.path.exists(h5_path):  #
        print(f"Warning: Residue embedding HDF5 file not found: {h5_path}. Cannot load vocabulary from it.")  #
        return None, None  #
    try:  #
        with h5py.File(h5_path, 'r') as hf:  #
            keys_in_file = list(hf.keys())  #
            for residue_key in keys_in_file:  # No tqdm here, usually small number of keys (e.g., 26) #
                if not isinstance(hf[residue_key], h5py.Dataset): continue  #
                vector = hf[residue_key][:]  #
                if not (vector.ndim == 1 and vector.shape[0] > 0): continue  #
                current_key_dim = vector.shape[0]  #
                if embedding_dim is None:  #
                    embedding_dim = current_key_dim  #
                elif embedding_dim != current_key_dim:  #
                    print(  #
                        f"  Warning: Inconsistent embedding dim for residue key '{residue_key}' in {os.path.basename(h5_path)}. Expected {embedding_dim}, got {current_key_dim}. Skipping this key.")  #
                    continue  #
                residue_vocab[residue_key] = vector.astype(np.float32)  #
    except Exception as e:  #
        print(f"  Error reading vocabulary HDF5 file {h5_path}: {e}")  #
        return None, None  # Indicate failure to load this vocab #

    if not residue_vocab:  #
        print(f"Warning: Residue embedding vocabulary from {h5_path} is empty.")  #
        return None, None  #
    return residue_vocab, embedding_dim  #


def pool_sequence_embeddings_worker(args: Tuple[str, str, Dict[str, np.ndarray], int, str]) -> Optional[  #
    Tuple[str, np.ndarray, Set[str]]]:  #
    prot_id, sequence, residue_embedding_vocab, embedding_dim, pooling_strategy = args  #
    if not sequence or embedding_dim == 0: return prot_id, None, set()  #
    sum_of_vectors = np.zeros(embedding_dim, dtype=np.float32)  #
    max_vector = np.full(embedding_dim, -np.inf, dtype=np.float32)  #
    valid_residues_count = 0  #
    unknown_residues_in_seq = set()  #
    for residue_char in sequence:  #
        ascii_key = str(ord(residue_char))  #
        vec = residue_embedding_vocab.get(ascii_key)  #
        if vec is not None:  #
            sum_of_vectors += vec  #
            max_vector = np.maximum(max_vector, vec)  #
            valid_residues_count += 1  #
        else:  #
            unknown_residues_in_seq.add(residue_char)  #
    if valid_residues_count == 0: return prot_id, None, unknown_residues_in_seq  #
    pooled_vector = None  #
    if pooling_strategy == 'mean':  #
        pooled_vector = sum_of_vectors / valid_residues_count  #
    elif pooling_strategy == 'sum':  #
        pooled_vector = sum_of_vectors  #
    elif pooling_strategy == 'max':  #
        pooled_vector = max_vector  #
    else:  #
        pooled_vector = sum_of_vectors / valid_residues_count  #
    return prot_id, pooled_vector.astype(np.float32) if pooled_vector is not None else None, unknown_residues_in_seq  #


def main():  #
    script_start_time = time.time()  #
    print(  #
        f"--- Script: Generating Per-Protein Embeddings (using ID Map) (Tag: {RUN_TAG}) ---")  #

    # Load the ID Mapping
    id_map = load_id_mapping(ID_MAPPING_FILE_PATH, MAPPING_FILE_FROM_ID_COLUMN, MAPPING_FILE_TO_ID_COLUMN)
    if not id_map and os.path.exists(ID_MAPPING_FILE_PATH):  # If map is empty after trying to load an existing file
        print(
            "CRITICAL: ID mapping is empty despite mapping file existing. Check mapping file format and column names.")
        return
    elif not id_map:  # If file didn't exist and map is empty (warning already printed by load_id_mapping)
        print("Continuing without ID mapping. IDs will be based on direct FASTA header parsing.")
        # In this case, you might want to fall back to the old parse_fasta_sequences or ensure
        # parse_fasta_sequences_with_mapping handles id_map being empty gracefully.
        # For now, parse_fasta_sequences_with_mapping will try to use it, and if an ID is not found, it's skipped.

    # ... (rest of main function as in your provided script, but ensure parse_fasta_sequences_with_mapping is called)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)  #
    if not os.path.isdir(OUTPUT_BASE_DIR): print(f"Error: Output dir {OUTPUT_BASE_DIR} error. Exiting."); return  #

    norm_input_h5_dir = os.path.normpath(INPUT_PER_RESIDUE_H5_DIR)  #
    if not os.path.isdir(norm_input_h5_dir): print(  #
        f"Error: Input HDF5 dir for residue embeddings not found: {norm_input_h5_dir}. Exiting."); return  #
    input_h5_file_paths = [p for ext in ('*.h5', '*.hdf5') for p in glob.glob(os.path.join(norm_input_h5_dir, ext))]  #
    if not input_h5_file_paths: print(f"No HDF5 files in {norm_input_h5_dir}. Exiting."); return  #
    print(f"Found {len(input_h5_file_paths)} HDF5 file(s) in {norm_input_h5_dir} to process individually.")  #

    norm_fasta_file = os.path.normpath(INPUT_FASTA_FILE)  #
    if not os.path.exists(norm_fasta_file): print(  #
        f"Error: Input FASTA file not found: {norm_fasta_file}. Exiting."); return  #

    fasta_entries_to_process: List[Tuple[str, str]]  #
    print(f"\nReading sequences from FASTA file: {norm_fasta_file}...")  #

    # Use the new parsing function with the loaded map
    all_entries_from_fasta = list(parse_fasta_sequences_with_mapping(norm_fasta_file, id_map))  #

    if not all_entries_from_fasta: print("No sequences reliably mapped from FASTA file. Exiting."); return  #

    if SAMPLE_N_SEQUENCES is not None and SAMPLE_N_SEQUENCES > 0:  #
        num_to_sample = SAMPLE_N_SEQUENCES  #
        if len(all_entries_from_fasta) < num_to_sample:  #
            print(  #
                f"Warning: Num sequences in FASTA ({len(all_entries_from_fasta)}) < sample size ({num_to_sample}). Processing all {len(all_entries_from_fasta)} seqs.")  #
            fasta_entries_to_process = all_entries_from_fasta  #
        else:  #
            print(f"Randomly sampling {num_to_sample} sequences from {len(all_entries_from_fasta)} total.")  #
            fasta_entries_to_process = random.sample(all_entries_from_fasta, num_to_sample)  #
    else:  #
        print(f"Processing all {len(all_entries_from_fasta)} sequences from FASTA file.")  #
        fasta_entries_to_process = all_entries_from_fasta  #

    if not fasta_entries_to_process: print("No protein sequences selected for processing. Exiting."); return  #

    # ... (The rest of your main() function continues here, no changes needed for the loop,
    #      worker calls, PCA, or saving, as they operate on the prot_id yielded by the parser)
    for h5_file_path in input_h5_file_paths:  #
        print(f"\n--- Processing proteins using vocabulary from: {os.path.basename(h5_file_path)} ---")  #

        residue_embedding_vocab, current_original_embedding_dim = load_single_h5_vocabulary(h5_file_path)  #
        if not residue_embedding_vocab or current_original_embedding_dim is None:  #
            print(f"  Could not load valid vocabulary from {h5_file_path}. Skipping this HDF5 file.")  #
            continue  #

        tasks_for_current_h5_vocab = [  #
            (prot_id, sequence, residue_embedding_vocab, current_original_embedding_dim, POOLING_STRATEGY)  #
            for prot_id, sequence in fasta_entries_to_process  #
        ]  #

        if not tasks_for_current_h5_vocab:  #
            print(  #
                f"  No tasks to process for HDF5: {os.path.basename(h5_file_path)} (possibly empty FASTA list after sampling).")  #
            continue  #

        final_pooled_embeddings_for_this_h5: Dict[str, np.ndarray] = {}  #
        global_unknown_residues_for_this_h5 = set()  #
        sequences_skipped_for_this_h5 = 0  #

        print(  #
            f"  Starting parallel pooling for {len(tasks_for_current_h5_vocab)} proteins using {os.path.basename(h5_file_path)} vocab...")  #
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSING_WORKERS) as executor:  #
            future_to_task_arg = {executor.submit(pool_sequence_embeddings_worker, task_arg): task_arg for task_arg in
                                  #
                                  tasks_for_current_h5_vocab}  #
            for future in tqdm(concurrent.futures.as_completed(future_to_task_arg),  #
                               total=len(tasks_for_current_h5_vocab),  #
                               desc=f"Pooling with {os.path.basename(h5_file_path)}"):  #
                prot_id_task = future_to_task_arg[future][0]  #
                try:  #
                    prot_id_result, pooled_vector, unknown_in_seq = future.result()  #
                    if pooled_vector is not None:  #
                        final_pooled_embeddings_for_this_h5[prot_id_result] = pooled_vector  #
                    else:  #
                        sequences_skipped_for_this_h5 += 1  #
                    if unknown_in_seq: global_unknown_residues_for_this_h5.update(unknown_in_seq)  #
                except Exception as exc:  #
                    print(  #
                        f"  Protein '{prot_id_task}' generated an exception: {exc}");
                    sequences_skipped_for_this_h5 += 1  #

        if global_unknown_residues_for_this_h5: print(  #
            f"  Warning: For vocab {os.path.basename(h5_file_path)}, residues not found: {sorted(list(global_unknown_residues_for_this_h5))}")  #
        if sequences_skipped_for_this_h5 > 0: print(  #
            f"  Warning: Skipped {sequences_skipped_for_this_h5} proteins for vocab {os.path.basename(h5_file_path)}.")  #
        if not final_pooled_embeddings_for_this_h5: print(  #
            f"  No per-protein embeddings generated using vocab {os.path.basename(h5_file_path)}. Skipping save for this file."); continue  #

        print(  #
            f"  Successfully generated {len(final_pooled_embeddings_for_this_h5)} pooled embeddings using {os.path.basename(h5_file_path)}.")  #

        current_embedding_dim_for_file = current_original_embedding_dim  #
        pca_applied_this_file = False  #
        if APPLY_PCA and current_embedding_dim_for_file > 0 and len(final_pooled_embeddings_for_this_h5) > 1:  #
            print(f"  Applying PCA (dim: {current_embedding_dim_for_file} -> {COMMON_EMBEDDING_DIM_PCA})...")  #
            ids_list_pca = list(final_pooled_embeddings_for_this_h5.keys())  #
            embeddings_array_pca = np.array([final_pooled_embeddings_for_this_h5[id_val] for id_val in ids_list_pca])  #
            if embeddings_array_pca.ndim == 1:  #
                try:  #
                    embeddings_array_pca = embeddings_array_pca.reshape(len(ids_list_pca),  #
                                                                        current_original_embedding_dim if current_original_embedding_dim else -1)  #
                except:  #
                    print(f"  Error reshaping for PCA. Skipping.")  #

            if embeddings_array_pca.ndim == 2 and embeddings_array_pca.shape[0] > 0 and embeddings_array_pca.shape[
                1] > 0:  #
                target_dim_pca = COMMON_EMBEDDING_DIM_PCA  #
                if embeddings_array_pca.shape[1] <= target_dim_pca and embeddings_array_pca.shape[
                    1] == COMMON_EMBEDDING_DIM_PCA:  #
                    print(f"    Embeddings already at or below target PCA dim {target_dim_pca}.")  #
                    if embeddings_array_pca.shape[1] == COMMON_EMBEDDING_DIM_PCA and embeddings_array_pca.shape[
                        1] > 0: pca_applied_this_file = True  #
                else:  #
                    max_pca_comp = min(embeddings_array_pca.shape[0] - 1 if embeddings_array_pca.shape[0] > 1 else 1,  #
                                       embeddings_array_pca.shape[1])  #
                    if max_pca_comp < 1: max_pca_comp = 1  #
                    pca_n_comp = min(target_dim_pca, max_pca_comp)  #
                    if pca_n_comp < 1 or pca_n_comp >= embeddings_array_pca.shape[1]:  #
                        print(  #
                            f"    Skipping PCA: n_components ({pca_n_comp}) invalid for reduction from {embeddings_array_pca.shape[1]}.")  #
                    else:  #
                        pca_instance = PCA(n_components=pca_n_comp, random_state=SEED)  #
                        try:  #
                            transformed = pca_instance.fit_transform(embeddings_array_pca)  #
                            print(f"    PCA applied. New dim: {transformed.shape[1]}")  #
                            current_embedding_dim_for_file = transformed.shape[1]  #
                            for i, id_val in enumerate(ids_list_pca): final_pooled_embeddings_for_this_h5[id_val] = \
                                transformed[i]  #
                            if current_embedding_dim_for_file == COMMON_EMBEDDING_DIM_PCA: pca_applied_this_file = True  #
                        except Exception as e_pca:  #
                            print(f"    Error during PCA: {e_pca}.")  #
            elif APPLY_PCA:  #
                print("    PCA: Array not suitable for PCA. Skipping.")  #
        elif APPLY_PCA and len(final_pooled_embeddings_for_this_h5) <= 1:  #
            print("    PCA skipped: Not enough samples (<=1).")  #

        input_h5_basename = os.path.splitext(os.path.basename(h5_file_path))[0]  #
        output_filename_tag = f"{RUN_TAG}_{input_h5_basename}"  #
        if pca_applied_this_file: output_filename_tag += f"_pca{COMMON_EMBEDDING_DIM_PCA}"  #

        final_output_h5_path = os.path.normpath(  #
            os.path.join(OUTPUT_BASE_DIR, f"{OUTPUT_FILENAME_PREFIX}_{output_filename_tag}.h5"))  #
        print(f"  Saving pooled embeddings (dim: {current_embedding_dim_for_file}) to {final_output_h5_path}...")  #
        try:  #
            os.makedirs(os.path.dirname(final_output_h5_path), exist_ok=True)  #
            with h5py.File(final_output_h5_path, 'w') as hf_out:  #
                for prot_id, emb_vec in final_pooled_embeddings_for_this_h5.items(): hf_out.create_dataset(prot_id,  #
                                                                                                           data=emb_vec)  #
                hf_out.attrs.update({  #
                    'embedding_type': f'per_protein_pooled_{POOLING_STRATEGY}',  #
                    'source_fasta_file': str(INPUT_FASTA_FILE),  #
                    'source_residue_embedding_H5_file': str(h5_file_path),  #
                    'original_residue_embedding_dim': current_original_embedding_dim if current_original_embedding_dim is not None else 'N/A',
                    #
                    'final_protein_vector_size': current_embedding_dim_for_file, 'run_tag': RUN_TAG})  #
                if pca_applied_this_file: hf_out.attrs['pca_applied_target_dim'] = COMMON_EMBEDDING_DIM_PCA  #
            print(  #
                f"  Successfully saved {len(final_pooled_embeddings_for_this_h5)} protein embeddings to {final_output_h5_path}")  #
        except Exception as e:  #
            print(f"  Error saving HDF5 file {final_output_h5_path}: {e}")  #

    gc.collect()  #
    print(f"\nScript finished in {time.time() - script_start_time:.2f} seconds.")  #


if __name__ == "__main__":  #
    main()  #
