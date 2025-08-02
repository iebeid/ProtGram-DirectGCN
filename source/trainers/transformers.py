# ==============================================================================
# MODULE: trainers/transformers.py
# PURPOSE: Generates per-protein embeddings using pre-trained Transformer
#          models from Hugging Face.
# VERSION: 4.5 (Final fix for TypeError by correcting the tf.function signature)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import time
from pathlib import Path
from typing import List, Dict, Mapping, Optional, Tuple

import tensorflow as tf
from tqdm.auto import tqdm
from transformers import AutoTokenizer, TFAutoModel, T5Tokenizer

from configuration.config import Config
from source.utils.data import DataUtils, FastaUtils
from source.utils.models import EmbeddingProcessor


class TransformerEmbedder:
    def __init__(self, config: Config):
        self.config = config
        DataUtils.print_header("TransformerEmbedder Initialized")

    @staticmethod
    def _get_model_inference_function(model: tf.keras.Model, is_t5: bool, use_xla: bool) -> tf.types.experimental.GenericFunction:
        """Creates a compiled TensorFlow function for faster inference."""
        print(f"  Creating inference function (is_t5={is_t5}, use_xla={use_xla})...")

        @tf.function
        def model_call(inputs_dict_tf):
            if is_t5:
                num_seqs = tf.shape(inputs_dict_tf['input_ids'])[0]
                decoder_start_id = model.config.decoder_start_token_id or 0
                decoder_input_ids = tf.fill((num_seqs, 1), tf.cast(decoder_start_id, inputs_dict_tf['input_ids'].dtype))
                # For T5, we explicitly do not pass token_type_ids
                return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'],
                             decoder_input_ids=decoder_input_ids)
            else:
                # BERT-like models accept token_type_ids
                return model(inputs_dict_tf)

        # --- MINIMAL & FINAL FIX: Correct the input_signature to match the tokenizer's output ---
        # The tokenizer for BERT produces 'token_type_ids', which was missing from our signature.
        # This flexible signature handles variable batch and sequence lengths.
        if is_t5:
            input_signature = {
                'input_ids': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=[None, None], dtype=tf.int32)
            }
        else:
            input_signature = {
                'input_ids': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                'attention_mask': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                'token_type_ids': tf.TensorSpec(shape=[None, None], dtype=tf.int32)  # This was the missing key
            }
        # --- END FIX ---

        concrete_function = model_call.get_concrete_function(input_signature)
        if use_xla:
            print("  JIT Compiling concrete function with XLA...")
            concrete_function = tf.function(concrete_function, jit_compile=True)

        return concrete_function

    def _generate_embeddings_for_single_model(self, model_config_item: Dict, all_sequences: List[Tuple[str, str]],
                                              id_map: Optional[Mapping]) -> Optional[str]:
        """Handles the full embedding generation pipeline for one transformer model."""
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

            inference_func = TransformerEmbedder._get_model_inference_function(model, is_t5,
                                                                               self.config.USE_XLA_COMPILATION)

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

            output_filename = f"{model_name}_{self.config.TRANSFORMER_POOLING_STRATEGY}_dim{embedding_dim_from_model}.h5"
            output_path = self.config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR / output_filename

            print(f"  Saving final embeddings to: {output_path}")
            if all_protein_embeddings:
                DataUtils.write_h5(all_protein_embeddings, output_path, f"Writing H5 for {model_name}")
                return str(output_path)
            else:
                print("  No final embeddings to save.")

        except Exception as e:
            print(f"\nFATAL ERROR during processing for model {model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            del model, tokenizer, inference_func, all_protein_embeddings
            gc.collect()
            if tf.executing_eagerly(): tf.keras.backend.clear_session()
            print(f"--- Finished Transformer: {model_name} ---")
        return None

    def run(self) -> Dict[str, Path]:
        """
        Main entry point for the Transformer embedding generation pipeline.
        """
        DataUtils.print_header("PIPELINE STEP: Generating Embeddings from Transformers")
        self.config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        generated_paths = {}

        if tf.config.list_physical_devices('GPU'):
            print("  TensorFlow: GPU available.")
        else:
            print("  TensorFlow: No GPU detected. Using CPU.")

        all_sequences = []
        for fasta_path in self.config.SEQUENCE_FILE_PATHS:
            all_sequences.extend(FastaUtils.parse_sequences([fasta_path]))

        if not all_sequences:
            print("Error: No sequences found in the configured FASTA files. Skipping.")
            return {}
        print(f"Found {len(all_sequences)} total sequences to process.")

        id_map = DataUtils.get_id_mapping(self.config)

        for model_config_item in self.config.TRANSFORMER_MODELS_TO_RUN:
            output_path = self._generate_embeddings_for_single_model(model_config_item, all_sequences, id_map)
            if output_path:
                generated_paths[model_config_item['name']] = Path(output_path)

        DataUtils.print_header("Transformer Embedding PIPELINE STEP FINISHED")
        return generated_paths
