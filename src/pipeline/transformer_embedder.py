# G:/My Drive/Knowledge/Research/TWU/Topics/AI in Proteomics/Protein-protein interaction prediction/Code/ProtDiGCN/src/pipeline/transformer_embedder.py
# ==============================================================================
# MODULE: pipeline/transformer_embedder.py
# PURPOSE: Generates per-protein embeddings using pre-trained Transformer
#          models from Hugging Face.
# VERSION: 3.0 (Refactored into TransformerEmbedderPipeline class)
# ==============================================================================

import os
import gc
import glob
from typing import List, Dict

import numpy as np
import h5py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel, T5Tokenizer
from tqdm.auto import tqdm

# Import from our new project structure
from src.config import Config
from src.utils.graph_utils import DirectedNgramGraphForGCN
from src.utils.data_utils import DataUtils, DataLoader, GroundTruthLoader
from src.utils.graph_utils import NgramGraph, DirectedNgramGraphForGCN
from src.utils.results_utils import EvaluationReporter
from src.utils.models_utils import EmbeddingLoader, EmbeddingProcessor
from src.models.protgram_directgcn import ProtNgramGCN


class TransformerEmbedderPipeline:
    """
    Orchestrates the generation of protein embeddings using pre-trained
    Transformer models.
    """

    def __init__(self, config: Config):
        """
        Initializes the Transformer embedding pipeline.

        Args:
            config (Config): The configuration object for the pipeline.
        """
        self.config = config

    @staticmethod
    def _get_model_inference_function(model: tf.keras.Model, is_t5: bool, use_xla: bool) -> tf.types.experimental.GenericFunction:
        """
        Creates a model inference function, optionally compiled with XLA for performance.
        Static method as it doesn't depend on instance state.
        """

        def model_call(inputs_dict_tf):
            if is_t5:
                num_seqs = tf.shape(inputs_dict_tf['input_ids'])[0]
                decoder_start_id = model.config.decoder_start_token_id or 0  # Handles None case
                decoder_input_ids = tf.fill((num_seqs, 1), tf.cast(decoder_start_id, inputs_dict_tf['input_ids'].dtype))
                return model(input_ids=inputs_dict_tf['input_ids'], attention_mask=inputs_dict_tf['attention_mask'], decoder_input_ids=decoder_input_ids)
            else:
                return model(inputs_dict_tf)

        if use_xla:
            return tf.function(model_call, jit_compile=True)
        return tf.function(model_call)

    def _generate_embeddings_for_single_model(self, model_config_item: Dict, fasta_files: List[str]):
        """
        Main workflow to generate and save embeddings for a single transformer model.
        Uses self.config for pipeline parameters.
        """
        model_name = model_config_item["name"]
        hf_id = model_config_item["hf_id"]
        is_t5 = model_config_item["is_t5"]
        # Use self.config for global transformer parameters
        batch_size = max(1, int(self.config.TRANSFORMER_BASE_BATCH_SIZE * model_config_item["batch_size_multiplier"]))

        DataUtils.print_header(f"Starting Transformer: {model_name} ({hf_id})")
        print(f"  Batch Size: {batch_size}, Max Length: {self.config.TRANSFORMER_MAX_LENGTH}, Pooling: {self.config.TRANSFORMER_POOLING_STRATEGY}")

        all_protein_embeddings = {}
        model, tokenizer, inference_func = None, None, None
        embedding_dim_from_model = 0

        try:
            print("  Loading tokenizer and model...")
            tokenizer_class = T5Tokenizer if is_t5 else AutoTokenizer
            tokenizer = tokenizer_class.from_pretrained(hf_id)
            model = TFAutoModel.from_pretrained(hf_id, from_pt=True)  # Assuming PyTorch checkpoints if TF not available
            inference_func = TransformerEmbedderPipeline._get_model_inference_function(model, is_t5, False)  # XLA can be a config option

            if hasattr(model.config, 'hidden_size'):
                embedding_dim_from_model = model.config.hidden_size
            elif hasattr(model.config, 'd_model'):
                embedding_dim_from_model = model.config.d_model
            else:
                print("  Warning: Could not directly determine embedding dimension from model config. Will infer from first batch.")
            print("  Model and tokenizer loaded.")

            for fasta_path in tqdm(fasta_files, desc=f"FASTA Files for {model_name}", leave=False):
                batch_sequences, batch_ids = [], []
                # Use DataLoader.parse_sequences for consistency
                for prot_id, sequence in DataLoader.parse_sequences(fasta_path):
                    if not sequence: continue
                    sequence = sequence.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
                    batch_sequences.append(" ".join(list(sequence)))  # Tokenizers often expect space-separated AAs
                    batch_ids.append(prot_id)

                    if len(batch_sequences) >= batch_size:
                        inputs = tokenizer(batch_sequences, padding="longest", truncation=True, return_tensors="tf", max_length=self.config.TRANSFORMER_MAX_LENGTH)
                        outputs = inference_func(inputs)
                        raw_batch_output = (outputs.encoder_last_hidden_state if is_t5 else outputs.last_hidden_state).numpy()

                        if embedding_dim_from_model == 0 and raw_batch_output.ndim == 3:
                            embedding_dim_from_model = raw_batch_output.shape[-1]
                            print(f"  Inferred embedding dimension: {embedding_dim_from_model}")

                        for i in range(len(batch_ids)):
                            seq_len = len(batch_sequences[i].replace(" ", ""))
                            residue_embeds = EmbeddingProcessor.extract_transformer_residue_embeddings(raw_batch_output[i], seq_len, is_t5)
                            if residue_embeds.size > 0:
                                pooled_vec = EmbeddingProcessor.pool_residue_embeddings(residue_embeds, self.config.TRANSFORMER_POOLING_STRATEGY, embedding_dim_if_empty=embedding_dim_from_model)
                                if pooled_vec.size > 0: all_protein_embeddings[batch_ids[i]] = pooled_vec
                        batch_sequences, batch_ids = [], []

                if batch_sequences:  # Process final leftover batch
                    inputs = tokenizer(batch_sequences, padding="longest", truncation=True, return_tensors="tf", max_length=self.config.TRANSFORMER_MAX_LENGTH)
                    outputs = inference_func(inputs)
                    raw_batch_output = (outputs.encoder_last_hidden_state if is_t5 else outputs.last_hidden_state).numpy()

                    if embedding_dim_from_model == 0 and raw_batch_output.ndim == 3:
                        embedding_dim_from_model = raw_batch_output.shape[-1]
                        print(f"  Inferred embedding dimension: {embedding_dim_from_model}")

                    for i in range(len(batch_ids)):
                        seq_len = len(batch_sequences[i].replace(" ", ""))
                        residue_embeds = EmbeddingProcessor.extract_transformer_residue_embeddings(raw_batch_output[i], seq_len, is_t5)
                        if residue_embeds.size > 0:
                            pooled_vec = EmbeddingProcessor.pool_residue_embeddings(residue_embeds, self.config.TRANSFORMER_POOLING_STRATEGY, embedding_dim_if_empty=embedding_dim_from_model)
                            if pooled_vec.size > 0: all_protein_embeddings[batch_ids[i]] = pooled_vec

            print(f"\n  Generated {len(all_protein_embeddings)} total protein embeddings for {model_name}.")

            final_embeddings = all_protein_embeddings
            if self.config.APPLY_PCA_TO_TRANSFORMER and len(all_protein_embeddings) > 1:
                print(f"  Applying PCA to {model_name} embeddings...")
                final_embeddings = EmbeddingProcessor.apply_pca(all_protein_embeddings, self.config.PCA_TARGET_DIMENSION, self.config.RANDOM_STATE)
                if final_embeddings is None:
                    print(f"  PCA failed for {model_name}, using full dimension embeddings.")
                    final_embeddings = all_protein_embeddings  # Revert to original if PCA fails

            output_filename_base = f"{model_name}_{self.config.TRANSFORMER_POOLING_STRATEGY}"

            # Determine if PCA was successfully applied and output dimension matches target
            pca_applied_successfully = False
            if self.config.APPLY_PCA_TO_TRANSFORMER and final_embeddings is not None:
                first_emb = next((v for v in final_embeddings.values() if v is not None and v.size > 0), None)
                if first_emb is not None and first_emb.shape[0] == self.config.PCA_TARGET_DIMENSION:
                    pca_applied_successfully = True

            if pca_applied_successfully:
                output_filename = f"{output_filename_base}_pca{self.config.PCA_TARGET_DIMENSION}.h5"
            else:
                # Use actual dimension if PCA was not applied or target dim wasn't met
                actual_dim = embedding_dim_from_model
                if final_embeddings:
                    first_emb_check = next((v for v in final_embeddings.values() if v is not None and v.size > 0), None)
                    if first_emb_check is not None:
                        actual_dim = first_emb_check.shape[0]
                output_filename = f"{output_filename_base}_dim{actual_dim}.h5"

            output_path = os.path.join(str(self.config.TRANSFORMER_EMBEDDINGS_DIR), output_filename)

            print(f"  Saving final embeddings to: {output_path}")
            if final_embeddings:
                with h5py.File(output_path, 'w') as hf:
                    for prot_id, embedding in final_embeddings.items():
                        if embedding is not None and embedding.size > 0:
                            hf.create_dataset(prot_id.replace('/', '_'), data=embedding)  # Sanitize ID for HDF5 key
                print("  Save complete.")
            else:
                print("  No final embeddings to save.")

        except Exception as e:
            print(f"\nFATAL ERROR during processing for model {model_name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"  Clearing {model_name} from memory...")
            del model, tokenizer, inference_func, all_protein_embeddings
            if 'final_embeddings' in locals(): del final_embeddings
            gc.collect()
            if tf.executing_eagerly():  # Check if in eager mode before clearing session
                tf.keras.backend.clear_session()
            print(f"--- Finished: {model_name} ---")

    def run_pipeline(self):
        """
        The main entry point for the Transformer embedding generation pipeline step.
        """
        DataUtils.print_header("PIPELINE STEP: Generating Embeddings from Transformers")
        os.makedirs(str(self.config.TRANSFORMER_EMBEDDINGS_DIR), exist_ok=True)

        if tf.config.list_physical_devices('GPU'):
            print("  TensorFlow: GPU available. Enabling Mixed Precision (mixed_float16).")
            try:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
            except Exception as e:
                print(f"  Warning: Could not enable mixed precision: {e}")
        else:
            print("  TensorFlow: No GPU detected by TensorFlow.")

        fasta_input_dir = str(self.config.TRANSFORMER_INPUT_FASTA_DIR)
        fasta_files = [os.path.normpath(f) for f in glob.glob(os.path.join(fasta_input_dir, '*.fasta')) + glob.glob(os.path.join(fasta_input_dir, '*.fa'))]
        if not fasta_files:
            print(f"Error: No FASTA files found in {fasta_input_dir}. Skipping Transformer embedding generation.")
            return
        print(f"Found {len(fasta_files)} FASTA file(s) to process from {fasta_input_dir}.")

        for model_config_item in self.config.TRANSFORMER_MODELS_TO_RUN:
            self._generate_embeddings_for_single_model(model_config_item, fasta_files)

        DataUtils.print_header("Transformer Embedding PIPELINE STEP FINISHED")
