# ==============================================================================
# MODULE: pipeline/transformer_embedder.py
# PURPOSE: Generates per-protein embeddings using pre-trained Transformer
#          models from Hugging Face.
# VERSION: 3.1 (Enhanced logging)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import glob
import os
import time
from typing import List, Dict

import h5py
import tensorflow as tf
from tqdm.auto import tqdm
from transformers import AutoTokenizer, TFAutoModel, T5Tokenizer

from config import Config
from src.utils.data_utils import DataUtils, DataLoader
from src.utils.models_utils import EmbeddingProcessor


class TransformerEmbedder:
    def __init__(self, config: Config):
        self.config = config
        print("TransformerEmbedder initialized.")
        DataUtils.print_header("TransformerEmbedder Initialized")


    @staticmethod
    def _get_model_inference_function(model: tf.keras.Model, is_t5: bool, use_xla: bool) -> tf.types.experimental.GenericFunction:
        print(f"  Creating inference function (is_t5={is_t5}, use_xla={use_xla})...")
        def model_call(inputs_dict_tf):
            if is_t5:
                num_seqs = tf.shape(inputs_dict_tf['input_ids'])[0]
                decoder_start_id = model.config.decoder_start_token_id or 0
                decoder_input_ids = tf.fill((num_seqs, 1), tf.cast(decoder_start_id, inputs_dict_tf['input_ids'].dtype))
                return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'], decoder_input_ids=decoder_input_ids)
            else:
                return model(inputs_dict_tf)

        if use_xla:
            print("  Compiling inference function with XLA.")
            return tf.function(model_call, jit_compile=True)
        return tf.function(model_call)

    def _generate_embeddings_for_single_model(self, model_config_item: Dict, fasta_files: List[str]):
        model_name = model_config_item["name"]
        hf_id = model_config_item["hf_id"]
        is_t5 = model_config_item["is_t5"]
        batch_size_multiplier = model_config_item.get("batch_size_multiplier", 1.0)
        batch_size = max(1, int(self.config.TRANSFORMER_BASE_BATCH_SIZE * batch_size_multiplier))

        DataUtils.print_header(f"Starting Transformer Embedding Generation: {model_name} ({hf_id})")
        print(
            f"  Config: Batch Size={batch_size} (Base: {self.config.TRANSFORMER_BASE_BATCH_SIZE}, Multiplier: {batch_size_multiplier}), Max Length={self.config.TRANSFORMER_MAX_LENGTH}, Pooling='{self.config.TRANSFORMER_POOLING_STRATEGY}', PCA={self.config.APPLY_PCA_TO_TRANSFORMER}")

        all_protein_embeddings = {}
        model, tokenizer, inference_func = None, None, None
        embedding_dim_from_model = 0
        model_load_start_time = time.time()

        try:
            print("  Loading tokenizer and model...")
            tokenizer_class = T5Tokenizer if is_t5 else AutoTokenizer
            tokenizer = tokenizer_class.from_pretrained(hf_id)
            model = TFAutoModel.from_pretrained(hf_id, from_pt=True)
            inference_func = TransformerEmbedder._get_model_inference_function(model, is_t5, False)

            if hasattr(model.config, 'hidden_size'):
                embedding_dim_from_model = model.config.hidden_size
            elif hasattr(model.config, 'd_model'):
                embedding_dim_from_model = model.config.d_model
            else:
                print("  Warning: Could not directly determine embedding dimension from model config. Will infer from first batch.")
            print(f"  Model and tokenizer loaded in {time.time() - model_load_start_time:.2f}s. Initial embedding_dim: {embedding_dim_from_model if embedding_dim_from_model > 0 else 'to be inferred'}")

            total_proteins_processed = 0
            for fasta_idx, fasta_path in enumerate(fasta_files):
                print(f"\n  Processing FASTA file {fasta_idx + 1}/{len(fasta_files)}: {os.path.basename(fasta_path)}")
                file_protein_count = 0
                batch_sequences, batch_ids = [], []

                for prot_id, sequence in tqdm(DataLoader.parse_sequences(fasta_path), desc=f"  Sequences in {os.path.basename(fasta_path)}", leave=False):
                    if not sequence:
                        print(f"    Skipping empty sequence for ID: {prot_id}")
                        continue

                    sequence = sequence.upper().replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
                    batch_sequences.append(" ".join(list(sequence)))
                    batch_ids.append(prot_id)
                    file_protein_count += 1

                    if len(batch_sequences) >= batch_size:
                        inputs = tokenizer(batch_sequences, padding="longest", truncation=True, return_tensors="tf", max_length=self.config.TRANSFORMER_MAX_LENGTH)
                        outputs = inference_func(inputs)
                        raw_batch_output = (outputs.encoder_last_hidden_state if is_t5 else outputs.last_hidden_state).numpy()

                        if embedding_dim_from_model == 0 and raw_batch_output.ndim == 3:
                            embedding_dim_from_model = raw_batch_output.shape[-1]
                            print(f"  Inferred embedding dimension from first batch: {embedding_dim_from_model}")

                        for i in range(len(batch_ids)):
                            seq_len_original = len(batch_sequences[i].replace(" ", ""))
                            residue_embeds = EmbeddingProcessor.extract_transformer_residue_embeddings(raw_batch_output[i], seq_len_original, is_t5)
                            if residue_embeds.size > 0:
                                pooled_vec = EmbeddingProcessor.pool_residue_embeddings(residue_embeds, self.config.TRANSFORMER_POOLING_STRATEGY, embedding_dim_if_empty=embedding_dim_from_model)
                                if pooled_vec.size > 0: all_protein_embeddings[batch_ids[i]] = pooled_vec
                        batch_sequences, batch_ids = [], []

                if batch_sequences:
                    inputs = tokenizer(batch_sequences, padding="longest", truncation=True, return_tensors="tf", max_length=self.config.TRANSFORMER_MAX_LENGTH)
                    outputs = inference_func(inputs)
                    raw_batch_output = (outputs.encoder_last_hidden_state if is_t5 else outputs.last_hidden_state).numpy()
                    if embedding_dim_from_model == 0 and raw_batch_output.ndim == 3:
                        embedding_dim_from_model = raw_batch_output.shape[-1]
                        print(f"  Inferred embedding dimension from final batch: {embedding_dim_from_model}")

                    for i in range(len(batch_ids)):
                        seq_len_original = len(batch_sequences[i].replace(" ", ""))
                        residue_embeds = EmbeddingProcessor.extract_transformer_residue_embeddings(raw_batch_output[i], seq_len_original, is_t5)
                        if residue_embeds.size > 0:
                            pooled_vec = EmbeddingProcessor.pool_residue_embeddings(residue_embeds, self.config.TRANSFORMER_POOLING_STRATEGY, embedding_dim_if_empty=embedding_dim_from_model)
                            if pooled_vec.size > 0: all_protein_embeddings[batch_ids[i]] = pooled_vec

                print(f"    Processed {file_protein_count} proteins from {os.path.basename(fasta_path)}.")
                total_proteins_processed += file_protein_count

            print(f"\n  Generated {len(all_protein_embeddings)} total protein embeddings for {model_name} from {total_proteins_processed} sequences.")

            final_embeddings_to_save = all_protein_embeddings
            output_filename_suffix = f"_dim{embedding_dim_from_model}"

            if self.config.APPLY_PCA_TO_TRANSFORMER and len(all_protein_embeddings) > self.config.PCA_TARGET_DIMENSION :
                print(f"  Applying PCA to {model_name} embeddings (target dim: {self.config.PCA_TARGET_DIMENSION})...")
                pca_start_time = time.time()
                pca_embeddings = EmbeddingProcessor.apply_pca(all_protein_embeddings, self.config.PCA_TARGET_DIMENSION, self.config.RANDOM_STATE)
                if pca_embeddings is not None:
                    final_embeddings_to_save = pca_embeddings
                    output_filename_suffix = f"_pca{self.config.PCA_TARGET_DIMENSION}"
                    print(f"  PCA applied in {time.time() - pca_start_time:.2f}s. New embedding dim: {self.config.PCA_TARGET_DIMENSION}")
                else:
                    print(f"  PCA failed or was skipped for {model_name}, using full dimension embeddings.")
            elif self.config.APPLY_PCA_TO_TRANSFORMER:
                print(f"  Skipping PCA for {model_name}: not enough samples ({len(all_protein_embeddings)}) for target dimension ({self.config.PCA_TARGET_DIMENSION}).")

            output_filename_base = f"{model_name}_{self.config.TRANSFORMER_POOLING_STRATEGY}"
            output_filename = f"{output_filename_base}{output_filename_suffix}.h5"
            output_path = os.path.join(str(self.config.TRANSFORMER_EMBEDDINGS_DIR), output_filename)

            print(f"  Saving final embeddings to: {output_path}")
            if final_embeddings_to_save:
                save_start_time = time.time()
                with h5py.File(output_path, 'w') as hf:
                    for prot_id, embedding in final_embeddings_to_save.items():
                        if embedding is not None and embedding.size > 0:
                            hf.create_dataset(prot_id.replace('/', '_'), data=embedding)
                print(f"  Save complete in {time.time() - save_start_time:.2f}s. Saved {len(final_embeddings_to_save)} embeddings.")
            else:
                print("  No final embeddings to save.")

        except Exception as e:
            print(f"\nFATAL ERROR during processing for model {model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"  Clearing {model_name} model and related objects from memory...")
            del model, tokenizer, inference_func, all_protein_embeddings
            if 'final_embeddings_to_save' in locals(): del final_embeddings_to_save
            if 'pca_embeddings' in locals() and pca_embeddings is not None: del pca_embeddings
            gc.collect()
            if tf.executing_eagerly():
                tf.keras.backend.clear_session()
            print(f"--- Finished Transformer: {model_name} ---")

    def run(self):
        DataUtils.print_header("PIPELINE STEP: Generating Embeddings from Transformers")
        os.makedirs(str(self.config.TRANSFORMER_EMBEDDINGS_DIR), exist_ok=True)

        if tf.config.list_physical_devices('GPU'):
            print("  TensorFlow: GPU available.")
        else:
            print("  TensorFlow: No GPU detected by TensorFlow. Using CPU.")

        fasta_input_dir = str(self.config.TRANSFORMER_INPUT_FASTA_DIR)
        fasta_files = sorted([os.path.normpath(f) for f in glob.glob(os.path.join(fasta_input_dir, '*.fasta')) + glob.glob(os.path.join(fasta_input_dir, '*.fa'))])

        if not fasta_files:
            print(f"Error: No FASTA files found in '{fasta_input_dir}'. Skipping Transformer embedding generation.")
            return
        print(f"Found {len(fasta_files)} FASTA file(s) to process from '{fasta_input_dir}': {fasta_files}")

        for model_config_item in self.config.TRANSFORMER_MODELS_TO_RUN:
            self._generate_embeddings_for_single_model(model_config_item, fasta_files)

        DataUtils.print_header("Transformer Embedding PIPELINE STEP FINISHED")
