# ==============================================================================
# MODULE: trainers/transformers.py
# PURPOSE: Generates per-protein embeddings using pre-trained Transformer
#          models from Hugging Face.
# VERSION: 4.0 (Added consistent protein ID mapping)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
import os
import time
from contextlib import nullcontext
from typing import List, Dict, Mapping, Optional, Union
from pathlib import Path

import h5py
import tensorflow as tf
from tqdm.auto import tqdm
from transformers import AutoTokenizer, TFAutoModel, T5Tokenizer

from configuration.config import Config
from source.utils.data import DataUtils, IDMapGenerator, FastaUtils
from source.utils.models import EmbeddingProcessor


class TransformerEmbedder:
    def __init__(self, config: Config):
        self.config = config
        DataUtils.print_header("TransformerEmbedder Initialized")

    def _load_id_map(self) -> Optional[Mapping]:
        """Loads the UniProt ID mapping file if configured."""
        if getattr(self.config, 'ID_MAPPING_MODE', 'none') != 'none':
            print("  Loading Protein ID Mapping for consistency...")
            id_mapper_instance = IDMapGenerator(config=self.config)
            id_map_result = id_mapper_instance.generate_id_maps()
            print(f"  ID mapping result of type '{type(id_map_result)}' loaded.")
            return id_map_result
        return None

    @staticmethod
    def _get_model_inference_function(model: tf.keras.Model, is_t5: bool, use_xla: bool) -> tf.types.experimental.GenericFunction:
        """Creates a compiled TensorFlow function for faster inference."""
        print(f"  Creating inference function (is_t5={is_t5}, use_xla={use_xla})...")

        def model_call(inputs_dict_tf):
            if is_t5:
                # T5 models require a decoder_input_ids argument for the encoder-decoder architecture
                num_seqs = tf.shape(inputs_dict_tf['input_ids'])[0]
                decoder_start_id = model.config.decoder_start_token_id or 0
                decoder_input_ids = tf.fill((num_seqs, 1), tf.cast(decoder_start_id, inputs_dict_tf['input_ids'].dtype))
                return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'],
                             decoder_input_ids=decoder_input_ids)
            else:
                # Standard BERT-like models
                return model(inputs_dict_tf)

        if use_xla:
            print("  Compiling inference function with XLA for potential performance boost.")
            return tf.function(model_call, jit_compile=True)
        return tf.function(model_call)

    def _generate_embeddings_for_single_model(self, model_config_item: Dict, fasta_files: List[Path],
                                              id_map: Optional[Mapping]) -> Optional[str]:
        """Handles the full embedding generation pipeline for one transformer model."""
        model_name = model_config_item["name"]
        hf_id = model_config_item["hf_id"]
        is_t5 = model_config_item["is_t5"]
        batch_size_multiplier = model_config_item.get("batch_size_multiplier", 1.0)
        batch_size = max(1, int(self.config.TRANSFORMER_BASE_BATCH_SIZE * batch_size_multiplier))

        DataUtils.print_header(f"Starting Transformer Embedding Generation: {model_name} ({hf_id})")
        print(
            f"  Config: Batch Size={batch_size}, Max Length={self.config.TRANSFORMER_MAX_LENGTH}, Pooling='{self.config.TRANSFORMER_POOLING_STRATEGY}', PCA={self.config.APPLY_PCA_TO_TRANSFORMER}")

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

            total_proteins_processed = 0
            for fasta_idx, fasta_path in enumerate(fasta_files):
                print(f"\n  Processing FASTA file {fasta_idx + 1}/{len(fasta_files)}: {fasta_path.name}")
                file_protein_count = 0
                batch_sequences, batch_ids = [], []
                for prot_id, sequence in tqdm(FastaUtils.parse_sequences([fasta_path]),
                                              desc=f"  Sequences in {fasta_path.name}", leave=False):
                    if not sequence: continue
                    sequence = sequence.upper().replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
                    batch_sequences.append(" ".join(list(sequence)))
                    batch_ids.append(prot_id)
                    file_protein_count += 1

                    if len(batch_sequences) >= batch_size:
                        inputs = tokenizer(batch_sequences, padding="longest", truncation=True, return_tensors="tf",
                                           max_length=self.config.TRANSFORMER_MAX_LENGTH)
                        outputs = inference_func(inputs)
                        raw_batch_output = (
                            outputs.encoder_last_hidden_state if is_t5 else outputs.last_hidden_state).numpy()
                        for i in range(len(batch_ids)):
                            seq_len_original = len(batch_sequences[i].replace(" ", ""))
                            residue_embeds = EmbeddingProcessor.extract_transformer_residue_embeddings(
                                raw_batch_output[i], seq_len_original, is_t5)
                            if residue_embeds.size > 0:
                                pooled_vec = EmbeddingProcessor.pool_residue_embeddings(residue_embeds,
                                                                                        self.config.TRANSFORMER_POOLING_STRATEGY,
                                                                                        embedding_dim_from_model)
                                if pooled_vec.size > 0: all_protein_embeddings[batch_ids[i]] = pooled_vec
                        batch_sequences, batch_ids = [], []

                if batch_sequences: # Process the final, potentially incomplete batch
                    inputs = tokenizer(batch_sequences, padding="longest", truncation=True, return_tensors="tf",
                                       max_length=self.config.TRANSFORMER_MAX_LENGTH)
                    outputs = inference_func(inputs)
                    raw_batch_output = (
                        outputs.encoder_last_hidden_state if is_t5 else outputs.last_hidden_state).numpy()
                    for i in range(len(batch_ids)):
                        seq_len_original = len(batch_sequences[i].replace(" ", ""))
                        residue_embeds = EmbeddingProcessor.extract_transformer_residue_embeddings(
                            raw_batch_output[i], seq_len_original, is_t5)
                        if residue_embeds.size > 0:
                            pooled_vec = EmbeddingProcessor.pool_residue_embeddings(residue_embeds,
                                                                                    self.config.TRANSFORMER_POOLING_STRATEGY,
                                                                                    embedding_dim_from_model)
                            if pooled_vec.size > 0: all_protein_embeddings[batch_ids[i]] = pooled_vec
                total_proteins_processed += file_protein_count

            print(
                f"\n  Generated {len(all_protein_embeddings)} total protein embeddings for {model_name} from {total_proteins_processed} sequences.")

            # --- CONSISTENCY FIX: Apply ID mapping ---
            if id_map:
                print("  Applying ID mapping to generated embeddings...")
                mapped_embeddings = {id_map.get(k, k): v for k, v in all_protein_embeddings.items()}
                print(f"    Original count: {len(all_protein_embeddings)}, Mapped count: {len(mapped_embeddings)}")
                all_protein_embeddings = mapped_embeddings
            # --- END FIX ---

            final_embeddings_to_save = all_protein_embeddings
            output_filename_suffix = f"_dim{embedding_dim_from_model}"
            if self.config.APPLY_PCA_TO_TRANSFORMER and len(
                    all_protein_embeddings) > self.config.PCA_TARGET_DIMENSION:
                print(f"  Applying PCA to {model_name} embeddings (target dim: {self.config.PCA_TARGET_DIMENSION})...")
                pca_embeddings = EmbeddingProcessor.apply_pca(all_protein_embeddings,
                                                              self.config.PCA_TARGET_DIMENSION, self.config.RANDOM_STATE)
                if pca_embeddings is not None:
                    final_embeddings_to_save = pca_embeddings
                    output_filename_suffix = f"_pca{self.config.PCA_TARGET_DIMENSION}"
            elif self.config.APPLY_PCA_TO_TRANSFORMER:
                print(
                    f"  Skipping PCA for {model_name}: not enough samples ({len(all_protein_embeddings)}) for target dimension ({self.config.PCA_TARGET_DIMENSION}).")

            output_filename = f"{model_name}_{self.config.TRANSFORMER_POOLING_STRATEGY}{output_filename_suffix}.h5"
            output_path = self.config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR / output_filename

            print(f"  Saving final embeddings to: {output_path}")
            if final_embeddings_to_save:
                DataUtils.write_h5(final_embeddings_to_save, output_path, f"Writing H5 for {model_name}")
                return str(output_path)
            else:
                print("  No final embeddings to save.")

        except Exception as e:
            print(f"\nFATAL ERROR during processing for model {model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Explicitly clean up to free GPU memory
            del model, tokenizer, inference_func, all_protein_embeddings
            gc.collect()
            if tf.executing_eagerly(): tf.keras.backend.clear_session()
            print(f"--- Finished Transformer: {model_name} ---")
        return None

    def run(self) -> Dict[str, Path]:
        """
        Main entry point for the Transformer embedding generation pipeline.
        Iterates through all configured models and generates embeddings for each.
        """
        DataUtils.print_header("PIPELINE STEP: Generating Embeddings from Transformers")
        self.config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        generated_paths = {}

        if tf.config.list_physical_devices('GPU'):
            print("  TensorFlow: GPU available.")
        else:
            print("  TensorFlow: No GPU detected. Using CPU.")

        fasta_files = self.config.SEQUENCE_FILE_PATHS
        if not fasta_files:
            print("Error: No FASTA files specified in 'config.SEQUENCE_FILE_PATHS'. Skipping.")
            return {}
        print(f"Found {len(fasta_files)} FASTA file(s) to process: {[p.name for p in fasta_files]}")

        id_map = self._load_id_map()
        # Use a context manager if the ID map is the on-disk SQLite version
        context = id_map if isinstance(id_map, IDMapGenerator) else nullcontext(id_map)

        with context as mapper:
            for model_config_item in self.config.TRANSFORMER_MODELS_TO_RUN:
                output_path = self._generate_embeddings_for_single_model(model_config_item, fasta_files, mapper)
                if output_path:
                    generated_paths[model_config_item['name']] = Path(output_path)

        DataUtils.print_header("Transformer Embedding PIPELINE STEP FINISHED")
        return generated_paths