# ==============================================================================
# MODULE: data_builders/build_graphs.py
# PURPOSE: A standalone entry point for running the graph building process.
#          This script is called as a subprocess by the main pipeline to
#          isolate the memory-intensive Dask operations.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import argparse
import sys
from pathlib import Path

# --- Add project root to sys.path to allow for relative imports ---
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from configuration.config import Config
from source.data_builders.protgram import ProtGramDataBuilder


def main():
    """
    Initializes the config and runs the appropriate graph builder.
    Accepts an optional FASTA file path to override the config, which is
    critical when this script is called as a subprocess.
    """
    parser = argparse.ArgumentParser(description="Run the ProtGram graph builder.")
    parser.add_argument(
        "--fasta_path",
        type=str,
        help="Optional path to a specific FASTA file to process, overriding the config."
    )
    args = parser.parse_args()

    config = Config()
    if args.fasta_path:
        print(f"  [build_graphs.py] Overriding config with FASTA path from command line: {args.fasta_path}")
        config.SEQUENCE_FILE_PATHS = [Path(args.fasta_path)]

    ProtGramDataBuilder(config).run()


if __name__ == "__main__":
    main()
