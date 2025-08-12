# ==============================================================================
# MODULE: trainers/transformers.py
# PURPOSE: Generates per-protein embeddings using pre-trained Transformer
#          models from Hugging Face.
# VERSION: 4.5 (Final fix for TypeError by correcting the tf.function signature)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Dict, Mapping, Optional, Tuple

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
        # Determine model type for signature generation
        is_esm_model = 'esm' in hf_id.lower()
        print(f"  Creating inference function (is_t5={is_t5}, is_esm={is_esm_model}, use_xla={use_xla})...")

        @tf.function
        def model_call(inputs_dict_tf):
            if is_t5:
                num_seqs = tf.shape(inputs_dict_tf['input_ids'])[0]
                decoder_start_id = model.config.decoder_start_token_id or 0
                decoder_input_ids = tf.fill((num_seqs, 1), tf.cast(decoder_start_id, inputs_dict_tf['input_ids'].dtype))
                # For T5, we explicitly do not pass token_type_ids
                return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'],
                             decoder_input_ids=decoder_input_ids)
            elif is_esm_model:
                # ESM models do not accept token_type_ids
                return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'])
            else:
                # Standard BERT-like models accept the full dictionary
                return model(inputs_dict_tf)

        if is_t5:
            input_signature = {
                'input_ids': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=[None, None], dtype=tf.int32)
            }
        elif is_esm_model:
            # ESM models do not use token_type_ids
            input_signature = {
                'input_ids': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=[None, None], dtype=tf.int32)
            }
        else:
            # BERT-like models use all three
            input_signature = {
                'input_ids': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                'token_type_ids': tf.TensorSpec(shape=[None, None], dtype=tf.int32)  # This was the missing key
            }

        concrete_function = model_call.get_concrete_function(input_signature)
        if use_xla:
            print("  JIT Compiling concrete function with XLA...")
            concrete_function = tf.function(concrete_function, jit_compile=True)

        return concrete_function

    def _generate_embeddings_for_single_model(self, model_config_item: Dict, all_sequences: List[Tuple[str, str]],
                                              id_map: Optional[Mapping]) -> Dict[str, np.ndarray]:
        """Handles the embedding generation for a single chunk of sequences for one transformer model."""
        model_name = model_config_item["name"]
        hf_id = model_config_item["hf_id"]
        is_t5 = model_config_item["is_t5"]
        batch_size_multiplier = model_config_item.get("batch_size_multiplier", 1.0)
        batch_size = max(1, int(self.config.TRANSFORMER_BASE_BATCH_SIZE * batch_size_multiplier))

        DataUtils.print_header(f"Starting Transformer Embedding Generation: {model_name} ({hf_id})")
        print(f"  Config: Batch Size={batch_size}, Max Length={self.config.TRANSFORMER_MAX_LENGTH}, Pooling='{self.config.TRANSFORMER_POOLING_STRATEGY}'")

        all_protein_embeddings = {}
        model, tokenizer, inference_func = None, None, None
        embedding_dim_from_model = 0
        model_load_start_time = time.time()

        try:
            print("  Loading tokenizer and model...")
            tokenizer_class = T5Tokenizer if is_t5 else AutoTokenizer
            tokenizer = tokenizer_class.from_pretrained(hf_id)
            model = TFAutoModel.from_pretrained(hf_id, from_pt=True)

            inference_func = TransformerEmbedder._get_model_inference_function(
                model, hf_id, is_t5, self.config.USE_XLA_COMPILATION)

            if hasattr(model.config, 'hidden_size'):
                embedding_dim_from_model = model.config.hidden_size
            elif hasattr(model.config, 'd_model'):
                embedding_dim_from_model = model.config.d_model
            else:
                print("  Warning: Could not determine embedding dimension from model config.")
            print(
                f"  Model and tokenizer loaded in {time.time() - model_load_start_time:.2f}s. Embedding dim: {embedding_dim_from_model}")

            print("  Sorting sequences by length for memory-efficient batching...")
            sorted_sequences = sorted(all_sequences, key=lambda x: len(x[1]))

            for i in tqdm(range(0, len(sorted_sequences), batch_size), desc=f"  Generating {model_name} Embeddings"):
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
                        pooled_vec = EmbeddingProcessor.pool_residue_embeddings(residue_embeds,
                                                                                self.config.TRANSFORMER_POOLING_STRATEGY,
                                                                                embedding_dim_from_model)
                        if pooled_vec.size > 0:
                            all_protein_embeddings[batch_ids[j]] = pooled_vec

            print(
                f"\n  Generated {len(all_protein_embeddings)} total protein embeddings for {model_name} from {len(sorted_sequences)} sequences.")

            if id_map:
                print("  Applying ID mapping to generated embeddings...")
                mapped_embeddings = {id_map.get(k, k): v for k, v in all_protein_embeddings.items()}
                print(f"    Original count: {len(all_protein_embeddings)}, Mapped count: {len(mapped_embeddings)}")
                all_protein_embeddings = mapped_embeddings

            return all_protein_embeddings

        except Exception as e:
            print(f"\nFATAL ERROR during processing for model {model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del model, tokenizer, inference_func, all_protein_embeddings
            gc.collect()
            if tf.executing_eagerly(): tf.keras.backend.clear_session()
            print(f"--- Finished processing chunk for Transformer: {model_name} ---")
        return {}

    def run(self) -> Dict[str, Path]:
        """
        Main entry point for the Transformer embedding generation pipeline.
        This method efficiently processes the sequence file in chunks, running all
        configured models on each chunk before loading the next one to minimize disk I/O.
        """
        DataUtils.print_header("PIPELINE STEP: Generating Embeddings from Transformers")
        self.config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        generated_paths = {}

        # --- NEW: Create a parent MLflow run for the entire Transformer pipeline ---
        mlflow_active = self.config.USE_MLFLOW
        run_context = mlflow.start_run(run_name="Transformer_Embedding_Pipeline") if mlflow_active else nullcontext()

        with run_context:
            if tf.config.list_physical_devices('GPU'):
                print("  TensorFlow: GPU available.")
            else:
                print("  TensorFlow: No GPU detected. Using CPU.")

            id_map = DataUtils.get_id_mapping(self.config)

            # Initialize a dictionary to hold the aggregated embeddings for each model.
            final_embeddings_per_model = {
                model_config['name']: {} for model_config in self.config.TRANSFORMER_MODELS_TO_RUN
            }

            # --- Main Efficient Loop ---
            # This loop reads the FASTA file only ONCE.
            chunk_size = getattr(self.config, 'TRANSFORMER_CHUNK_SIZE', 10000)
            sequence_iterator = FastaUtils.parse_sequences(self.config.SEQUENCE_FILE_PATHS)
            chunk_num = 0
            while True:
                chunk_num += 1
                # Read one chunk of sequences into memory
                chunk = [item for _, item in zip(range(chunk_size), sequence_iterator)]
                if not chunk:
                    break

                DataUtils.print_header(f"Processing Sequence Chunk {chunk_num} ({len(chunk)} sequences)")

                # Iterate through each configured model and process the SAME chunk
                for model_config_item in self.config.TRANSFORMER_MODELS_TO_RUN:
                    model_name = model_config_item['name']
                    chunk_embeddings = self._generate_embeddings_for_single_model(model_config_item, chunk, id_map)
                    if chunk_embeddings:
                        final_embeddings_per_model[model_name].update(chunk_embeddings)

            # --- Save final results after all chunks have been processed ---
            DataUtils.print_header("Saving All Transformer Embeddings")
            model_configs_by_name = {mc['name']: mc for mc in self.config.TRANSFORMER_MODELS_TO_RUN}
            for model_name, embeddings_dict in final_embeddings_per_model.items():
                if embeddings_dict:
                    # --- NEW: Create a nested MLflow run for this specific model ---
                    nested_run_context = mlflow.start_run(run_name=model_name, nested=True) if mlflow_active else nullcontext()
                    with nested_run_context:
                        model_config = model_configs_by_name.get(model_name)
                        if mlflow_active and model_config:
                            mlflow.log_params({
                                "hf_id": model_config['hf_id'],
                                "is_t5": model_config['is_t5'],
                                "pooling_strategy": self.config.TRANSFORMER_POOLING_STRATEGY
                            })

                        embedding_dim = next(iter(embeddings_dict.values())).shape[0]
                        output_filename = f"{model_name}_{self.config.TRANSFORMER_POOLING_STRATEGY}_dim{embedding_dim}.h5"
                        output_path = self.config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR / output_filename
                        print(f"  Saving final aggregated embeddings for {model_name}...")
                        DataUtils.write_h5(embeddings_dict, output_path, f"Writing H5 for {model_name}")
                        generated_paths[model_name] = output_path

                        if mlflow_active:
                            mlflow.log_artifact(str(output_path), "final_embeddings")
                else:
                    print(f"  No embeddings were generated for {model_name}. Skipping save.")

        DataUtils.print_header("Transformer Embedding PIPELINE STEP FINISHED")
        return generated_paths