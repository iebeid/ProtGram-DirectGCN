# ==============================================================================
# MODULE: utils/model_converter.py
# PURPOSE: A standalone utility to convert Hugging Face models to a local
#          TensorFlow format, acting as a disk cache.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import argparse
import sys
import traceback
from pathlib import Path

# --- FIX: Add the project root to the Python path to allow relative imports ---
# This must be done BEFORE any local modules (like 'configuration') are imported.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from configuration.config import Config
# --- Local import to avoid circular dependency at module level ---
from transformers import AutoTokenizer, TFAutoModel


class ModelConverter:
    """
    A utility class for handling model format conversions.
    """

    @staticmethod
    def convert_and_save_model(model_id: str, output_base_dir: Path):
        """
        Converts a PyTorch model from Hugging Face to TensorFlow format and saves it locally.
        This is a memory-intensive, one-time operation to prevent OOM errors in the main pipeline.
        It checks for existence before running the conversion.
        """
        print(f"--- Checking for local pre-converted model: {model_id} ---")

        output_path = output_base_dir / model_id
        output_path.mkdir(parents=True, exist_ok=True)

        # Check if the model is already converted and saved
        if (output_path / "tf_model.h5").exists() and (output_path / "config.json").exists():
            print(f"  ✅ TensorFlow model already exists locally. Skipping conversion.")
            return

        print(f"  Local model not found. Starting conversion process...")
        print(f"  Output will be saved to: {output_path}")

        try:
            # --- REFACTOR: Load the tokenizer once, as it's the same for both cases ---
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Check if the model requires conversion in the first place
            try:
                print("  Attempting to download native TensorFlow weights directly...")
                model = TFAutoModel.from_pretrained(model_id)
                print("  Native TF weights found. Saving them locally...")
            except OSError as e:
                if "from_pt=True" in str(e):
                    print("  Native TF weights not found. Converting from PyTorch...")
                    print("  !!! THIS STEP IS MEMORY-INTENSIVE AND MAY TAKE A WHILE !!!")
                    model = TFAutoModel.from_pretrained(model_id, from_pt=True)
                    print("  Model converted to TensorFlow in memory.")
                else:
                    raise e

            print("\n  Saving model and tokenizer to disk...")
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"\n--- SUCCESS: Model is now available locally at: {output_path} ---")

        except Exception as e:
            print(f"\n--- ❌ An error occurred during model conversion/saving for '{model_id}': {e} ---")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="A standalone script to convert a Hugging Face model to TensorFlow format."
    )
    parser.add_argument(
        "model_id",
        type=str,
        help="The Hugging Face model identifier (e.g., 'Rostlab/prot_bert')."
    )
    args = parser.parse_args()

    config = Config()
    ModelConverter.convert_and_save_model(
        model_id=args.model_id,
        output_base_dir=config.DATA_MODELS_DIR
    )


if __name__ == "__main__":
    main()