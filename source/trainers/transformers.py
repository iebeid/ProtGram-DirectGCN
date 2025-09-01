# ==============================================================================
# MODULE: trainers/transformers.py
# PURPOSE: Generates per-protein embeddings using pre-trained Transformer
#          models from Hugging Face.
# VERSION: 7.0 (Refactored to use centralized IDMapper singleton)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================

import gc
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
import mlflow
import random
from typing import Dict, Optional, Tuple, Any, Mapping
import torch
import mlflow
import tensorflow as tf
from tqdm.auto import tqdm
from tf_keras import mixed_precision
from transformers import AutoTokenizer, TFAutoModel, T5Tokenizer
import numpy as np
from configuration.config import Config
from source.utils.data.data_utils import DataUtils
from source.utils.data.fasta_utils import FastaUtils
from source.utils.fs.file_utils import FileUtils
from source.utils.data.id_mapper import IDMapper
from source.utils.post.embedding_processor import EmbeddingProcessor


class TransformerEmbedder:
    def __init__(self, config: Config):
        self.config = config
        DataUtils.print_header("TransformerEmbedder Initialized")

    @staticmethod
    def _get_model_inference_function(model: tf.keras.Model, hf_id: str, is_t5: bool, use_xla: bool) -> tf.types.experimental.GenericFunction:
        """Creates a compiled TensorFlow function for faster inference."""
        is_esm_model = 'esm' in hf_id.lower()
        print(f"  Creating inference function (is_t5={is_t5}, is_esm={is_esm_model}, use_xla={use_xla})...")

        @tf.function(reduce_retracing=True)
        def model_call(inputs_dict_tf):
            if is_t5:
                num_seqs = tf.shape(inputs_dict_tf['input_ids'])[0]
                decoder_start_id = model.config.decoder_start_token_id or 0
                decoder_input_ids = tf.fill((num_seqs, 1), tf.cast(decoder_start_id, inputs_dict_tf['input_ids'].dtype))
                return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'],
                             decoder_input_ids=decoder_input_ids)
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

        # The policy is now handled at a higher level in _load_transformer_model
        concrete_function = model_call.get_concrete_function(input_signature)

        # XLA is disabled for ESM models due to the same underlying type incompatibility.
        if use_xla and not is_esm_model:
            print("  JIT Compiling concrete function with XLA...")
            concrete_function = tf.function(concrete_function, jit_compile=True)

        return concrete_function

    def _load_transformer_model(self, model_config_item: Dict) -> Tuple[Optional[tf.keras.Model], Optional[Any], Optional[tf.types.experimental.GenericFunction], int]:
        """
        Loads a single transformer model, tokenizer, and creates an inference function.
        This helper centralizes the model loading logic with robust error handling.
        """
        model_name = model_config_item["name"]
        hf_id = model_config_item["hf_id"]
        is_t5 = model_config_item["is_t5"]
        is_esm_model = 'esm' in hf_id.lower()

        DataUtils.print_header(f"Loading Transformer Model: {model_name} ({hf_id})")
        model_load_start_time = time.time()

        # --- DEFINITIVE FIX for ESM Model TypeError: Manage policy during loading ---
        # The ESM model must be loaded and compiled while the global policy is float32.
        # We wrap the entire process in a try/finally block to ensure the original
        # mixed_float16 policy is always restored for other models.
        original_policy = None
        if is_esm_model:
            print("  Temporarily setting policy to float32 for ESM model loading and compilation...")
            original_policy = mixed_precision.global_policy()
            mixed_precision.set_global_policy('float32')

        try:
            local_model_path = self.config.DATA_MODELS_DIR / hf_id
            if local_model_path.exists() and (local_model_path / "tf_model.h5").exists():
                print(f"  Found locally pre-converted TensorFlow model at: {local_model_path}")
                print("  Loading from local path...")
                tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                model = TFAutoModel.from_pretrained(local_model_path)
            else:
                print("  No local model found. Downloading from Hugging Face Hub...")
                tokenizer = AutoTokenizer.from_pretrained(hf_id)
                model = TFAutoModel.from_pretrained(hf_id)

            if model is None or tokenizer is None:
                raise RuntimeError("Model or tokenizer could not be loaded. The pre-conversion step may have failed.")

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
        finally:
            # Restore the original policy if it was changed
            if original_policy:
                print("  Restoring original mixed precision policy...")
                mixed_precision.set_global_policy(original_policy)

    def run(self) -> Dict[str, Path]:
        """
        Main entry point for the Transformer embedding generation pipeline.
        """
        mlflow.set_experiment(self.config.MLFLOW_LLMS_EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"Transformers_{Path(self.config.SEQUENCE_FILE_PATHS[0]).stem}"):
            DataUtils.print_header("PIPELINE STEP: Generating Embeddings from Transformers")
            self.config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            generated_paths: Dict[str, Path] = {}
            _model_cache: Dict[str, Tuple] = {}

            all_sequences = list(FastaUtils.parse_sequences(self.config.SEQUENCE_FILE_PATHS))
            sample_fraction = getattr(self.config, 'TRANSFORMER_INFERENCE_SAMPLE_FRACTION', 1.0)

            if 0.0 < sample_fraction < 1.0:
                num_to_sample = int(len(all_sequences) * sample_fraction)
                random.seed(self.config.RANDOM_STATE)
                sequences_to_process = random.sample(all_sequences, num_to_sample)
                print(f"  INFO: Using a random sample of {len(sequences_to_process)} sequences ({sample_fraction:.1%}) for Transformer inference.")
            else:
                sequences_to_process = all_sequences
            del all_sequences

            try:
                for model_config_item in self.config.TRANSFORMER_MODELS_TO_RUN:
                    model_name = model_config_item["name"]
                    hf_id = model_config_item["hf_id"]
                    all_protein_embeddings_for_model = {}

                    try:
                        with mlflow.start_run(run_name=model_name, nested=True):
                            mlflow.log_params(model_config_item)
                            DataUtils.print_header(f"Starting Transformer Embedding Generation: {model_name} ({hf_id})")
                            if hf_id in _model_cache:
                                model, tokenizer, inference_func, embedding_dim = _model_cache[hf_id]
                            else:
                                model, tokenizer, inference_func, embedding_dim = self._load_transformer_model(model_config_item)
                                if model and tokenizer and inference_func:
                                    _model_cache[hf_id] = (model, tokenizer, inference_func, embedding_dim)

                            if not model or not tokenizer or not inference_func:
                                continue

                            batch_size_multiplier = model_config_item.get('batch_size_multiplier', 1.0)
                            effective_batch_size = int(self.config.TRANSFORMER_BASE_BATCH_SIZE * batch_size_multiplier)
                            sorted_sequences = sorted(sequences_to_process, key=lambda x: len(x[1]))

                            with tqdm(total=len(sorted_sequences), desc=f"  Processing {model_name}") as pbar:
                                for i in range(0, len(sorted_sequences), effective_batch_size):
                                    batch = sorted_sequences[i:i + effective_batch_size]
                                    batch_ids = [seq[0] for seq in batch]
                                    batch_sequences = [seq[1] for seq in batch]

                                    inputs = tokenizer(
                                        batch_sequences, return_tensors="tf", padding="longest",
                                        truncation=True, max_length=self.config.TRANSFORMER_MAX_LENGTH
                                    )
                                    model_output = inference_func(inputs)
                                    raw_embeddings = model_output.last_hidden_state.numpy()

                                    for j, prot_id in enumerate(batch_ids):
                                        original_seq_len = len(batch_sequences[j])
                                        residue_embeddings = EmbeddingProcessor.extract_transformer_residue_embeddings(
                                            raw_model_output=raw_embeddings[j],
                                            original_sequence_length=original_seq_len,
                                            is_t5_model=model_config_item["is_t5"]
                                        )
                                        pooled_embedding = EmbeddingProcessor.pool_residue_embeddings(
                                            residue_embeddings,
                                            strategy=self.config.TRANSFORMER_POOLING_STRATEGY,
                                            embedding_dim_if_empty=embedding_dim
                                        )
                                        all_protein_embeddings_for_model[prot_id] = pooled_embedding.astype(np.float16)

                                    pbar.update(len(batch))

                            output_filename = f"{model_name}_{self.config.TRANSFORMER_POOLING_STRATEGY}_dim{embedding_dim}.h5"
                            output_path = self.config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR / output_filename
                            FileUtils.write_h5(all_protein_embeddings_for_model, output_path, f"Writing H5 for {model_name}")
                            generated_paths[model_name] = output_path

                    except Exception as e:
                        print(f"\n--- ❌ ERROR processing model {model_name}: {e} ---")
                        traceback.print_exc()
                    finally:
                        if hf_id in _model_cache:
                            del _model_cache[hf_id]
                        if 'model' in locals(): del model
                        if 'tokenizer' in locals(): del tokenizer
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            finally:
                del _model_cache
                gc.collect()

            DataUtils.print_header("Transformer Embedding PIPELINE STEP FINISHED")
            return generated_paths