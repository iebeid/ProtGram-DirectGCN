# ==============================================================================
# MODULE: testers/unit_tests.py
# PURPOSE: A unified and refactored script for all environment checks,
#          unit tests, and pipeline smoke tests for the ProtGram-DirectGCN project.
# VERSION: 2.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import unittest
from .gpu import GPUTests
from .data import DataUtilityTests, GraphBuilderTests
from .models import ModelBuildTests
from .reporting import ReportingTests
from .word2vec import Word2VecPipelineTests
from .transformers import TransformerPipelineTests
from .gnns import GNNBenchmarkerTests
from .ppi import PPIPipelineTests


def run_all_tests():
    """Main function to run all tests."""
    gpu_ok = GPUTests.verify_full_gpu_environment()

    # Create a TestSuite
    suite = unittest.TestSuite()

    # Add tests from each class
    suite.addTest(unittest.makeSuite(DataUtilityTests))
    suite.addTest(unittest.makeSuite(GraphBuilderTests))
    suite.addTest(unittest.makeSuite(ModelBuildTests))
    suite.addTest(unittest.makeSuite(ReportingTests))
    suite.addTest(unittest.makeSuite(Word2VecPipelineTests))
    suite.addTest(unittest.makeSuite(TransformerPipelineTests))
    suite.addTest(unittest.makeSuite(GNNBenchmarkerTests))
    suite.addTest(unittest.makeSuite(PPIPipelineTests))

    # Run the tests
    runner = unittest.TextTestRunner()
    runner.run(suite)

    return gpu_ok


if __name__ == "__main__":
    run_all_tests()