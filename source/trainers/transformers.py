# ==============================================================================
# MODULE: trainers/transformers.py
# PURPOSE: Generates per-protein embeddings using pre-trained Transformer
#          models from Hugging Face.
# VERSION: 5.0 (Refactored for efficiency and correctness)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import gc
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import List, Dict, Mapping, Optional, Tuple, Any

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
import mlflow
from transformers import AutoTokenizer, TFAutoModel, T5Tokenizer

from configuration.config import Config
from source.utils.data import DataUtils, FastaUtils
from source.utils.models import EmbeddingProcessor


class TransformerEmbedder:
    def __init__(self, config: Config):
        self.config = config
        DataUtils.print_header("TransformerEmbedder Initialized")

    @staticmethod
    def _get_model_inference_function(model: tf.keras.Model, hf_id: str, is_t5: bool, use_xla: bool) -> tf.types.experimental.GenericFunction:
        """Creates a compiled TensorFlow function for faster inference."""
        is_esm_model = 'esm' in hf_id.lower()
        print(f"  Creating inference function (is_t5={is_t5}, is_esm={is_esm_model}, use_xla={use_xla})...")

        # --- FIX: Add reduce_retracing=True to prevent excessive and slow re-compilations for variable sequence lengths ---
        @tf.function(reduce_retracing=True)
        def model_call(inputs_dict_tf):
            if is_t5:
                num_seqs = tf.shape(inputs_dict_tf['input_ids'])[0]
                decoder_start_id = model.config.decoder_start_token_id or 0
                decoder_input_ids = tf.fill((num_seqs, 1), tf.cast(decoder_start_id, inputs_dict_tf['input_ids'].dtype))
                return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'],
                             decoder_input_ids=decoder_input_ids)
            elif is_esm_model:
                return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'])
            else:
                return model(inputs_dict_tf)

        if is_t5 or is_esm_model:
            input_signature = {
                'input_ids': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=[None, None], dtype=tf.int32)
            }
        else:  # Standard BERT-like models
            input_signature = {
                'input_ids': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                'token_type_ids': tf.TensorSpec(shape=[None, None], dtype=tf.int32)
            }

        concrete_function = model_call.get_concrete_function(input_signature)
        if use_xla:
            print("  JIT Compiling concrete function with XLA...")
            concrete_function = tf.function(concrete_function, jit_compile=True)

        return concrete_function

    def _load_transformer_model(self, model_config_item: Dict) -> Tuple[Optional[tf.keras.Model], Optional[Any], Optional[tf.types.experimental.GenericFunction], int]:
        """
        Loads a single transformer model, tokenizer, and creates an inference function.
        This helper centralizes the model loading logic.
        """
        model_name = model_config_item["name"]
        hf_id = model_config_item["hf_id"]
        is_t5 = model_config_item["is_t5"]

        DataUtils.print_header(f"Loading Transformer Model: {model_name} ({hf_id})")
        model_load_start_time = time.time()

        try:
            tokenizer_class = T5Tokenizer if is_t5 else AutoTokenizer
            tokenizer = tokenizer_class.from_pretrained(hf_id)

        # --- DEFINITIVE FIX for model loading and memory issues ---
        # First, try to load the native TensorFlow model. This is the most memory-efficient.
        try:
            print("  Attempting to load native TensorFlow weights...")
            model = TFAutoModel.from_pretrained(hf_id)
        except OSError as e:
            # If native TF weights are not found, the library helpfully tells us to use `from_pt=True`.
            if "from_pt=True" in str(e):
                print("  Native TF weights not found. Falling back to loading from PyTorch weights (`from_pt=True`).")
                print("  NOTE: This can be memory-intensive for large models and may fail on systems with limited RAM.")
                model = TFAutoModel.from_pretrained(hf_id, from_pt=True)
            else:
                raise e # Re-raise other OSErrors (e.g., network issues)

        inference_func = self._get_model_inference_function(
            model, hf_id, is_t5, self.config.USE_XLA_COMPILATION)

        if hasattr(model.config, 'hidden_size'):
            embedding_dim = model.config.hidden_size
        elif hasattr(model.config, 'd_model'):
            embedding_dim = model.config.d_model
        else:
            embedding_dim = 0
            print("  Warning: Could not determine embedding dimension from model config.")

        print(f"  Model and tokenizer loaded in {time.time() - model_load_start_time:.2f}s. Embedding dim: {embedding_dim}")
        return model, tokenizer, inference_func, embedding_dim

    except Exception as e:
        print(f"\nFATAL ERROR during model loading for {model_name}: {e}")
        traceback.print_exc()
        return None, None, None, 0

    def run(self) -> Dict[str, Path]:
        """
        Main entry point for the Transformer embedding generation pipeline.
        This method efficiently processes the sequence file by loading each model
        only once and then iterating through chunks of data.
        """
        DataUtils.print_header("PIPELINE STEP: Generating Embeddings from Transformers")
        self.config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        generated_paths = {}

        mlflow_active = self.config.USE_MLFLOW
        run_context = mlflow.start_run(run_name="Transformer_Embedding_Pipeline") if mlflow_active else nullcontext()

        with run_context:
            if tf.config.list_physical_devices('GPU'):
                print("  TensorFlow: GPU available.")
            else:
                print("  TensorFlow: No GPU detected. Using CPU.")

            id_map = DataUtils.get_id_mapping(self.config)

            # --- Main Efficient Loop: Iterate through MODELS first ---
            for model_config_item in self.config.TRANSFORMER_MODELS_TO_RUN:
                model_name = model_config_item["name"]
                hf_id = model_config_item["hf_id"]
                is_t5 = model_config_item["is_t5"]
                batch_size_multiplier = model_config_item.get("batch_size_multiplier", 1.0)
                batch_size = max(1, int(self.config.TRANSFORMER_BASE_BATCH_SIZE * batch_size_multiplier))

                DataUtils.print_header(f"Starting Transformer Embedding Generation: {model_name} ({hf_id})")
                print(
                    f"  Config: Batch Size={batch_size}, Max Length={self.config.TRANSFORMER_MAX_LENGTH}, Pooling='{self.config.TRANSFORMER_POOLING_STRATEGY}'")

                model, tokenizer, inference_func = None, None, None
                all_protein_embeddings_for_model = {}

                try:
                    # --- Load model and tokenizer ONCE per model using the helper function ---
                    model, tokenizer, inference_func, embedding_dim = self._load_transformer_model(model_config_item)
                    if not model or not tokenizer or not inference_func:
                        continue  # Skip to the next model if loading failed

                    # --- Inner Loop: Iterate through DATA CHUNKS ---
                    sequence_iterator = FastaUtils.parse_sequences(self.config.SEQUENCE_FILE_PATHS)
                    chunk_num = 0
                    while True:
                        chunk_num += 1
                        chunk = [item for _, item in zip(range(self.config.TRANSFORMER_CHUNK_SIZE), sequence_iterator)]
                        if not chunk:
                            break

                        print(f"\n  Processing Sequence Chunk {chunk_num} ({len(chunk)} sequences) for model '{model_name}'...")
                        sorted_sequences = sorted(chunk, key=lambda x: len(x[1]))

                        for i in tqdm(range(0, len(sorted_sequences), batch_size), desc=f"    Generating Embeddings"):
                            batch = sorted_sequences[i:i + batch_size]
                            if not batch: continue

                            batch_ids = [item[0] for item in batch]
                            batch_sequences_text = [" ".join(list(item[1])) for item in batch]

                            inputs = tokenizer(
                                batch_sequences_text,
                                padding="longest",
                                truncation=True,
                                return_tensors="tf",
                                max_length=self.config.TRANSFORMER_MAX_LENGTH
                            )

                            outputs = inference_func(inputs)
                            raw_batch_output = (
                                outputs.encoder_last_hidden_state if is_t5 else outputs.last_hidden_state).numpy()

                            for j in range(len(batch_ids)):
                                seq_len_original = int(tf.reduce_sum(inputs['attention_mask'][j]))
                                residue_embeds = EmbeddingProcessor.extract_transformer_residue_embeddings(
                                    raw_batch_output[j], seq_len_original, is_t5)
                                if residue_embeds.size > 0:
                                    pooled_vec = EmbeddingProcessor.pool_residue_embeddings(
                                        residue_embeds,
                                        self.config.TRANSFORMER_POOLING_STRATEGY,
                                        embedding_dim
                                    )
                                    if pooled_vec.size > 0:
                                        all_protein_embeddings_for_model[batch_ids[j]] = pooled_vec

                    # --- After all chunks are processed for this model ---
                    if not all_protein_embeddings_for_model:
                        print(f"  No embeddings were generated for {model_name}. Skipping save.")
                        continue

                    print(
                        f"\n  Generated {len(all_protein_embeddings_for_model)} total protein embeddings for {model_name}.")

                    if id_map:
                        print("  Applying ID mapping to generated embeddings...")
                        mapped_embeddings = {id_map.get(k, k): v for k, v in
                                             all_protein_embeddings_for_model.items()}
                        print(
                            f"    Original count: {len(all_protein_embeddings_for_model)}, Mapped count: {len(mapped_embeddings)}")
                        all_protein_embeddings_for_model = mapped_embeddings

                    # Save the final aggregated embeddings for this model
                    nested_run_context = mlflow.start_run(run_name=model_name,
                                                          nested=True) if mlflow_active else nullcontext()
                    with nested_run_context:
                        output_filename = f"{model_name}_{self.config.TRANSFORMER_POOLING_STRATEGY}_dim{embedding_dim}.h5"
                        output_path = self.config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR / output_filename
                        print(f"  Saving final aggregated embeddings for {model_name}...")
                        DataUtils.write_h5(all_protein_embeddings_for_model, output_path,
                                           f"Writing H5 for {model_name}")
                        generated_paths[model_name] = output_path

                        if mlflow_active:
                            mlflow.log_params({
                                "hf_id": hf_id,
                                "is_t5": is_t5,
                                "pooling_strategy": self.config.TRANSFORMER_POOLING_STRATEGY
                            })
                            mlflow.log_artifact(str(output_path), "final_embeddings")

                except Exception as e:
                    print(f"\nFATAL ERROR during processing for model {model_name}: {e}")
                    traceback.print_exc()
                finally:
                    # Clean up memory before loading the next model
                    del model, tokenizer, inference_func, all_protein_embeddings_for_model
                    gc.collect()
                    if tf.executing_eagerly():
                        tf.keras.backend.clear_session()
                    print(f"--- Finished processing for Transformer: {model_name} ---")

        DataUtils.print_header("Transformer Embedding PIPELINE STEP FINISHED")
        return generated_paths