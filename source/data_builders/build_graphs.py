# ==============================================================================
# MODULE: data_builders/build_graphs.py
# PURPOSE: A standalone entry point for running the graph building process.
#          This script is called as a subprocess by the main pipeline to
#          isolate the memory-intensive Dask operations.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import sys
from pathlib import Path

# --- Add project root to sys.path to allow for relative imports ---
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from configuration.config import Config
from source.data_builders.fastprotgram import FastProtGramDataBuilder
from source.data_builders.protgram import ProtGramDataBuilder


def main():
    """
    Initializes the config and runs the appropriate graph builder.
    """
    config = Config()
    if config.USE_FAST_GRAPH_BUILDER:
        FastProtGramDataBuilder(config).run()
    else:
        ProtGramDataBuilder(config).run()


if __name__ == "__main__":
    main()