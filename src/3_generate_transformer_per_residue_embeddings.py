import os
import time
import re
import gc
import glob
from typing import List, Tuple, Dict, Iterator  # Added Iterator

import numpy as np
import h5py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, T5Tokenizer
from tqdm.auto import tqdm

# --- User Configuration ---
DATASET_TAG = "uniref50_transformers_per_residue"  # EXAMPLE: CHANGE THIS!
# DATASET_TAG = "dummy_transformers_per_residue"

# Input FASTA files
FASTA_INPUT_DIR = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/fasta/uniref50/"
FASTA_FILE_PATHS_LIST = None  # Or provide a list

# Output Directory for HDF5 files
# OUTPUT_EMBEDDINGS_DIR = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/transformer_per_residue_embeddings"
OUTPUT_EMBEDDINGS_DIR = "C:/tmp/Models/"


# Transformer Models to Use
MODEL_CONFIGS = [
    {"name": "ProtBERT", "hf_id": "Rostlab/prot_bert", "is_t5": False, "batch_size_multiplier": 1}
    # {"name": "ProtT5_XL", "hf_id": "Rostlab/prot_t5_xl_uniref50", "is_t5": True, "batch_size_multiplier": 0.25}
    # T5 XL is large, use smaller batch
]

# Processing Parameters
DEFAULT_MAX_LENGTH = 1024
BASE_BATCH_SIZE = 16  # Increased base batch size for better GPU utilization

# Performance Options
ENABLE_MIXED_PRECISION = True  # Set to True to try mixed precision
ENABLE_XLA = False  # Set to True to try XLA (can be model-dependent for compatibility)


# --- End User Configuration ---

# --- Helper Functions (from your models-delete.py) ---
def simple_fasta_parser(fasta_filepath: str) -> Iterator[Tuple[str, str]]:  # Copied from Script A
    current_id_full_header = None;
    uniprot_id = None;
    sequence_parts = []
    try:
        with open(fasta_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if current_id_full_header and sequence_parts: yield uniprot_id if uniprot_id else current_id_full_header, "".join(
                        sequence_parts)
                    current_id_full_header = line[1:]
                    parts = current_id_full_header.split('|')
                    uniprot_id = parts[1] if len(parts) > 1 and parts[1] else current_id_full_header.split()[0]
                    sequence_parts = []
                elif current_id_full_header:
                    sequence_parts.append(line.upper())
            if current_id_full_header and sequence_parts: yield uniprot_id if uniprot_id else current_id_full_header, "".join(
                sequence_parts)
    except FileNotFoundError:
        print(f"Warning: FASTA file not found: {fasta_filepath}")
    except Exception as e:
        print(f"Warning: Error parsing FASTA file {fasta_filepath}: {e}")


# --- Main Processing Function ---
def generate_embeddings_for_model(
        model_name_tag: str,  # e.g., "ProtBERT"
        huggingface_model_id: str,
        is_t5: bool,
        effective_batch_size: int,
        fasta_files: List[str],
        max_seq_len: int,
        output_h5_path: str
):
    print(f"\n--- Generating Per-Residue Embeddings with: {model_name_tag} ({huggingface_model_id}) ---")
    print(f"Effective batch size: {effective_batch_size}, Max length: {max_seq_len}")

    all_per_residue_embeddings_for_h5 = {}  # Store as {prot_id: np.array(seq_len, emb_dim)}
    native_embedding_dim = None

    try:
        print(f"Loading tokenizer for {huggingface_model_id}...")
        tokenizer = T5Tokenizer.from_pretrained(huggingface_model_id) if is_t5 else AutoTokenizer.from_pretrained(
            huggingface_model_id)
        print("Tokenizer loaded.")

        print(f"Loading model {huggingface_model_id} (from_pt=True)...")
        model = TFAutoModel.from_pretrained(huggingface_model_id, from_pt=True)
        print("Model loaded.")

        # For XLA (optional, can sometimes make first inference slow)
        if ENABLE_XLA:
            print("Attempting to wrap model inference with XLA-compiled tf.function...")

            @tf.function(jit_compile=True)
            def compiled_model_call(inputs_dict_tf):
                if is_t5:
                    num_seqs = tf.shape(inputs_dict_tf['input_ids'])[0]
                    decoder_start_id = model.config.decoder_start_token_id or model.config.pad_token_id or 0
                    decoder_input_ids_tf = tf.fill((num_seqs, 1),
                                                   tf.cast(decoder_start_id, inputs_dict_tf['input_ids'].dtype))
                    return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'],
                                 decoder_input_ids=decoder_input_ids_tf)
                else:
                    return model(inputs_dict_tf)

            model_inference_func = compiled_model_call
        else:
            def regular_model_call(inputs_dict_tf):
                if is_t5:
                    num_seqs = tf.shape(inputs_dict_tf['input_ids'])[0]
                    decoder_start_id = model.config.decoder_start_token_id or model.config.pad_token_id or 0
                    decoder_input_ids_tf = tf.fill((num_seqs, 1),
                                                   tf.cast(decoder_start_id, inputs_dict_tf['input_ids'].dtype))
                    return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'],
                                 decoder_input_ids=decoder_input_ids_tf)
                else:
                    return model(inputs_dict_tf)

            model_inference_func = regular_model_call


    except Exception as e:
        print(f"Error loading model or tokenizer for {model_name_tag}: {e}")
        return  # Skip this model if loading fails

    for fasta_file_path in tqdm(fasta_files, desc=f"Files for {model_name_tag}"):
        print(f"\n  Processing FASTA file: {os.path.basename(fasta_file_path)} with {model_name_tag}")
        sequences_batch_text = []
        identifiers_batch = []

        for prot_id, sequence in simple_fasta_parser(fasta_file_path):
            if not sequence: continue
            # ProtBERT/ProtT5 expect spaces between AAs for pre-tokenized inputs, or use raw string for AutoTokenizer
            # The AutoTokenizer usually handles adding spaces if needed based on model type.
            # Let's pass sequences as they are, with internal spaces removed if any.
            sequences_batch_text.append(re.sub(r"\s+", "", sequence))  # Remove internal spaces just in case
            identifiers_batch.append(prot_id)

            if len(sequences_batch_text) >= effective_batch_size:
                try:
                    # Tokenize (Hugging Face usually expects list of strings)
                    inputs_tf = tokenizer(sequences_batch_text, padding="max_length", truncation=True,
                                          return_tensors="tf", max_length=max_seq_len)

                    model_outputs = model_inference_func(inputs_tf)

                    raw_embeddings_batch_tf = model_outputs.encoder_last_hidden_state if is_t5 else model_outputs.last_hidden_state
                    raw_embeddings_batch_np = raw_embeddings_batch_tf.numpy()
                    attention_mask_batch_np = inputs_tf['attention_mask'].numpy()

                    if native_embedding_dim is None and raw_embeddings_batch_np.ndim == 3:
                        native_embedding_dim = raw_embeddings_batch_np.shape[2]

                    for i in range(len(identifiers_batch)):
                        seq_id = identifiers_batch[i]
                        true_len = int(np.sum(attention_mask_batch_np[i]))
                        # Store only the embeddings for actual residues (excluding padding, CLS, SEP if any)
                        # Rostlab models usually don't add CLS/SEP to per-residue outputs directly, it's residue-level
                        # For ProtBERT: embeddings are for [CLS] S1 S2 ... SN [SEP] Pad...
                        # For ProtT5: embeddings are for S1 S2 ... SN <EOS> Pad...
                        # We need to be careful about slicing.
                        # Assuming standard HuggingFace output, where first token can be CLS and last SEP for BERT-like.
                        # T5 output is usually just the residues.

                        # For simplicity and consistency with many ProtTrans examples:
                        # Assume the output `raw_embeddings_batch_np[i]` has shape (max_len, embed_dim)
                        # And we want the embeddings for the actual sequence part.
                        # The `attention_mask` helps identify true tokens.
                        # However, `true_len` from attention mask includes special tokens if tokenizer adds them.
                        # Most ProtTrans models give per-residue embeddings matching original sequence length after removing special tokens.

                        # Let's assume sequence length matches original for per-residue, and slicing if needed:
                        # Example: For ProtBERT, skip CLS [0] and SEP/PAD at end [true_len-1:]
                        # This depends heavily on tokenizer and model specifics.
                        # For now, let's save up to true_len if that's what Rostlab models provide as per-residue.
                        # Best practice: Check model documentation for precise output indexing.
                        # A common approach is to take embeddings from index 1 to true_len-1 for BERT if CLS/SEP are present.
                        # Or simply all tokens indicated by attention_mask, then map back.

                        # Sticking to a simple approach: save all tokens up to attention_mask length.
                        # The user's original average_pool used the full attention mask.
                        # For per-residue, the meaning of each token position matters.
                        # For Rostlab models, they usually state per-residue means for each AA.

                        # Simple approach: save all embeddings up to the sequence length (before padding)
                        # Tokenizer often adds special tokens. `true_len` from attention mask INCLUDES these.
                        # For per-residue, we want to map back to original amino acids.
                        # Sequence length of original sequence:
                        original_seq_len = len(sequences_batch_text[i])

                        # Slice based on original sequence length, assuming direct correspondence after special tokens
                        # This heuristic might need adjustment based on specific model/tokenizer
                        if is_t5:  # T5 usually doesn't have CLS, ends with EOS
                            embeddings_to_save = raw_embeddings_batch_np[i, :original_seq_len, :]
                        else:  # ProtBERT might have CLS at start, SEP at end
                            embeddings_to_save = raw_embeddings_batch_np[i, 1:original_seq_len + 1,
                                                 :]  # Heuristic: skip CLS

                        if embeddings_to_save.shape[0] == original_seq_len:  # Check if length matches
                            all_per_residue_embeddings_for_h5[seq_id] = embeddings_to_save.astype(
                                np.float16)  # Save as float16 for space
                        else:
                            # Fallback or warning if slicing heuristic is off
                            print(
                                f"Warning: Length mismatch for {seq_id}. Original: {original_seq_len}, Emb shape[0]: {embeddings_to_save.shape[0]}. Using available embeddings up to original_seq_len.")
                            # Save what we can, up to original length from the start of tensor
                            safe_len = min(original_seq_len, raw_embeddings_batch_np.shape[1])
                            all_per_residue_embeddings_for_h5[seq_id] = raw_embeddings_batch_np[i, :safe_len, :].astype(
                                np.float16)


                except Exception as batch_exc:
                    print(f"Error processing batch with {model_name_tag}: {batch_exc}")
                finally:
                    sequences_batch_text, identifiers_batch = [], []  # Reset batch

        # Process any remaining sequences
        if sequences_batch_text:
            try:
                inputs_tf = tokenizer(sequences_batch_text, padding="max_length", truncation=True, return_tensors="tf",
                                      max_length=max_seq_len)
                model_outputs = model_inference_func(inputs_tf)
                raw_embeddings_batch_tf = model_outputs.encoder_last_hidden_state if is_t5 else model_outputs.last_hidden_state
                raw_embeddings_batch_np = raw_embeddings_batch_tf.numpy()
                attention_mask_batch_np = inputs_tf['attention_mask'].numpy()
                if native_embedding_dim is None and raw_embeddings_batch_np.ndim == 3: native_embedding_dim = \
                raw_embeddings_batch_np.shape[2]
                for i in range(len(identifiers_batch)):
                    seq_id = identifiers_batch[i];
                    original_seq_len = len(sequences_batch_text[i])
                    embeddings_to_save = raw_embeddings_batch_np[i, 1:original_seq_len + 1,
                                         :] if not is_t5 else raw_embeddings_batch_np[i, :original_seq_len, :]
                    if embeddings_to_save.shape[0] == original_seq_len:
                        all_per_residue_embeddings_for_h5[seq_id] = embeddings_to_save.astype(np.float16)
                    else:
                        safe_len = min(original_seq_len, raw_embeddings_batch_np.shape[1])
                        all_per_residue_embeddings_for_h5[seq_id] = raw_embeddings_batch_np[i, :safe_len, :].astype(
                            np.float16)

            except Exception as batch_exc:
                print(f"Error processing final batch with {model_name_tag}: {batch_exc}")

    # Save to HDF5
    print(f"\nSaving per-residue embeddings from {model_name_tag} to {output_h5_path}...")
    try:

        output_h5_path = os.path.normpath(output_h5_path)
        print(f"Normalized HDF5 output path: {output_h5_path}")  # For debugging
        os.makedirs(os.path.dirname(output_h5_path),
                    exist_ok=True)  # Ensure directory exists after potential normalization

        with h5py.File(output_h5_path, 'w') as hf:
            if not all_per_residue_embeddings_for_h5:
                hf.attrs['status'] = f'No per-residue embeddings generated for {model_name_tag}'
            for prot_id, res_embs in all_per_residue_embeddings_for_h5.items():
                hf.create_dataset(prot_id, data=res_embs)  # Already float16
            hf.attrs['embedding_type'] = f'{model_name_tag}_per_residue'
            hf.attrs['original_vector_size'] = native_embedding_dim if native_embedding_dim is not None else -1
            hf.attrs['dataset_tag'] = DATASET_TAG
        print(f"Successfully saved per-residue embeddings for {model_name_tag} to {output_h5_path}")
    except Exception as e:
        print(f"Error saving HDF5 file for {model_name_tag} at {output_h5_path}: {e}")

    print(f"Clearing {model_name_tag} model from memory...")
    del model, tokenizer
    if 'model_inference_func' in locals(): del model_inference_func
    gc.collect()
    if tf.config.list_logical_devices('GPU'):  # Basic way to check if GPU was used by TF
        print(f"GPU memory should be clearer for {model_name_tag}.")
    print(f"{model_name_tag} processing complete and memory cleared.")


def main():
    print("--- Script C: Transformer Per-Residue Embedding Generation ---")
    os.makedirs(OUTPUT_EMBEDDINGS_DIR, exist_ok=True)

    if ENABLE_MIXED_PRECISION:
        print("Enabling Mixed Precision (mixed_float16) for TensorFlow.")
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        except Exception as e:
            print(f"Warning: Could not enable mixed precision: {e}")

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

    for config in MODEL_CONFIGS:
        model_tag_name = config["name"].replace(" ", "_")  # ProtBERT, ProtT5_XL
        output_h5_name = f"{model_tag_name}_per_residue_embeddings_{DATASET_TAG}.h5"
        output_h5_full_path = os.path.join(OUTPUT_EMBEDDINGS_DIR, output_h5_name)

        effective_batch_size = max(1, int(BASE_BATCH_SIZE * config["batch_size_multiplier"]))

        generate_embeddings_for_model(
            model_name_tag=config["name"],
            huggingface_model_id=config["hf_id"],
            is_t5=config["is_t5"],
            effective_batch_size=effective_batch_size,
            fasta_files=actual_fasta_files,
            max_seq_len=DEFAULT_MAX_LENGTH,
            output_h5_path=output_h5_full_path
        )
    print("Script C finished.")


if __name__ == "__main__":
    main()