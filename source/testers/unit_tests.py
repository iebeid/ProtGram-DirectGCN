# ==============================================================================
# MODULE: testers/unit_tests.py
# PURPOSE: A unified and refactored script for all environment checks,
#          unit tests, and pipeline smoke tests for the ProtGram-DirectGCN project.
# VERSION: 2.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import unittest
import argparse
from .gpu import GPUTests
from .data_utility import DataUtilityTests
from .graph_utility import GraphBuilderTests
from .models import ModelBuildTests
from .reporting import ReportingTests
from .word2vec import Word2VecPipelineTests
from .transformers import TransformerPipelineTests
from .gnns import GNNBenchmarkerTests
from .ppi import PPIPipelineTests


def run_all_tests(suites_to_run=None, verbosity=2):
    """Main function to run all tests."""
    gpu_ok = GPUTests.verify_full_gpu_environment()

    # --- REFACTOR: Allow selective test execution ---
    all_suites = {
        "data": DataUtilityTests,
        "graph": GraphBuilderTests,
        "models": ModelBuildTests,
        "reporting": ReportingTests,
        "w2v": Word2VecPipelineTests,
        "transformer": TransformerPipelineTests,
        "gnn": GNNBenchmarkerTests,
        "ppi": PPIPipelineTests
    }

    if suites_to_run is None or not suites_to_run:
        # If no specific suites are requested, run all of them.
        tests_to_load = all_suites.values()
        print("\n--- Running all test suites ---")
    else:
        # Otherwise, only run the requested suites.
        tests_to_load = [all_suites[name] for name in suites_to_run if name in all_suites]
        print(f"\n--- Running selected test suites: {', '.join(suites_to_run)} ---")

    suite = unittest.TestSuite()
    for test_class in tests_to_load:
        suite.addTest(unittest.makeSuite(test_class))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(suite)

    return gpu_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the project's test suite.")
    parser.add_argument(
        '--suite', nargs='*', choices=['data', 'graph', 'models', 'reporting', 'w2v', 'transformer', 'gnn', 'ppi'],
        help='Specify which test suites to run. If not provided, all suites will be run.'
    )
    parser.add_argument('--verbosity', type=int, default=2, help='Set the verbosity level for the test runner.')
    args = parser.parse_args()

    run_all_tests(suites_to_run=args.suite, verbosity=args.verbosity)