# ==============================================================================
# MODULE: trainers/transformers.py
# PURPOSE: Generates protein embeddings using pre-trained Transformer models.
# VERSION: 2.0 (Implemented sorting by length for OOM-safe batched inference)
# AUTHOR: Islam Ebeid
# ==============================================================================

import gc
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from transformers import AutoTokenizer, TFAutoModel

from configuration.config import Config
from source.utils.data import DataUtils, FastaUtils


class TransformerEmbedder:
    """
    Handles the generation of embeddings from pre-trained protein language models
    like ProtBERT, ProtT5, etc.
    """

    def __init__(self, config: Config):
        self.config = config
        self.output_dir = self.config.RESULTS_TRANSFORMER_EMBEDDINGS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = "GPU:0" if tf.config.list_physical_devices('GPU') else "CPU:0"
        print(f"  TensorFlow: {'GPU' if self.device == 'GPU:0' else 'CPU'} available.")

    def _load_model_and_tokenizer(self, model_info: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        """Loads the specified Hugging Face model and tokenizer."""
        hf_id = model_info["hf_id"]
        print(f"  Loading tokenizer and model for {hf_id}...")
        tokenizer = AutoTokenizer.from_pretrained(hf_id, do_lower_case=False)
        # Load model with from_pt=True to handle PyTorch checkpoints correctly
        model = TFAutoModel.from_pretrained(hf_id, from_pt=True)
        # Create a compiled inference function for a massive speedup
        inference_func = tf.function(
            model,
            jit_compile=self.config.USE_XLA_COMPILATION,
            input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                             tf.TensorSpec(shape=[None, None], dtype=tf.int32)]
        )
        return tokenizer, model, inference_func

    def _get_sequences_from_fasta(self) -> List[Tuple[str, str]]:
        """Loads all sequences from the configured FASTA files."""
        print(f"Found {len(self.config.SEQUENCE_FILE_PATHS)} FASTA file(s) to process.")
        all_sequences = []
        for file_path in self.config.SEQUENCE_FILE_PATHS:
            all_sequences.extend(list(FastaUtils.parse_sequences([file_path])))
        return all_sequences

    def run(self) -> Dict[str, Path]:
        """
        Main execution method to generate embeddings for all configured models.
        """
        DataUtils.print_header("PIPELINE STEP: Generating Embeddings from Transformers")
        all_sequences = self._get_sequences_from_fasta()
        id_map = DataUtils.get_id_mapping(self.config)
        generated_paths = {}

        for model_info in self.config.TRANSFORMER_MODELS_TO_RUN:
            model_name = model_info["name"]
            DataUtils.print_header(f"Starting Transformer Embedding Generation: {model_name} ({model_info['hf_id']})",
                                   level=2)

            tokenizer, model, inference_func = self._load_model_and_tokenizer(model_info)
            emb_dim = model.config.hidden_size
            print(f"  Model and tokenizer loaded. Embedding dim: {emb_dim}")

            protein_embeddings = {}
            batch_size = int(self.config.TRANSFORMER_BASE_BATCH_SIZE * model_info.get("batch_size_multiplier", 1))

            # --- OOM FIX: Sort sequences by length BEFORE batching ---
            # This groups sequences of similar lengths, minimizing memory waste from padding.
            print("  Sorting sequences by length for memory-efficient batching...")
            sorted_sequences = sorted(all_sequences, key=lambda x: len(x[1]))
            # --- END FIX ---

            # Iterate over the sorted list in batches
            for i in tqdm(range(0, len(sorted_sequences), batch_size), desc=f"  Generating {model_name} Embeddings"):
                batch = sorted_sequences[i:i + batch_size]
                if not batch: continue

                batch_ids = [item[0] for item in batch]
                batch_seqs_text = [" ".join(list(seq)) for _, seq in batch]

                inputs = tokenizer(
                    batch_seqs_text,
                    add_special_tokens=True,
                    padding="longest",  # Padding is now efficient due to sorting
                    truncation=True,
                    max_length=self.config.TRANSFORMER_MAX_LENGTH,
                    return_tensors="tf"
                )

                with tf.device(self.device):
                    outputs = inference_func(inputs['input_ids'], inputs['attention_mask'])

                # Detach from graph and move to CPU
                embeddings = outputs.last_hidden_state.numpy()

                for j in range(len(embeddings)):
                    seq_len = np.sum(inputs['attention_mask'][j].numpy())
                    # Mean pooling over the actual, non-padded sequence length
                    embedding = embeddings[j, :seq_len].mean(axis=0)
                    protein_embeddings[batch_ids[j]] = embedding

            print(f"  Generated {len(protein_embeddings)} total protein embeddings for {model_name}.")

            # Apply ID mapping and save
            mapped_embeddings = DataUtils.apply_id_mapping(protein_embeddings, id_map)
            final_embeddings = mapped_embeddings if mapped_embeddings else protein_embeddings

            # Handle PCA
            if self.config.APPLY_PCA_TO_TRANSFORMER:
                final_embeddings, emb_dim = DataUtils.apply_pca(
                    embeddings_dict=final_embeddings,
                    target_dim=self.config.PCA_TARGET_DIMENSION
                )
                suffix = f"mean_pca{emb_dim}"
            else:
                suffix = f"mean_dim{emb_dim}"

            output_filename = f"{model_name}_{suffix}.h5"
            output_path = self.output_dir / output_filename
            DataUtils.write_h5(final_embeddings, output_path, f"Writing H5 for {model_name}")
            generated_paths[f"{model_name}-Generated"] = output_path
            print(f"--- Finished Transformer: {model_name} ---")

            # Clean up memory
            del model, tokenizer, inference_func, protein_embeddings, final_embeddings
            gc.collect()

        DataUtils.print_header("Transformer Embedding PIPELINE STEP FINISHED", level=2)
        return generated_paths
