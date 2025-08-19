# ==============================================================================
# MODULE: testers/model_converter.py
# PURPOSE: Contains tests for model building and instantiation.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import unittest
from configuration.config import Config
from source.models.fnn.mlp import MLP
from source.utils.data.data_utils import DataUtils


class ModelBuildTests(unittest.TestCase):
    """A class for testing model building and instantiation."""

    def test_mlp_model_build(self):
        print("\n" + "=" * 80)
        DataUtils.print_header("MLP Model Build Test")
        print("=" * 80)
        config_instance = Config()
        mlp_params = {'dense1_units': 32, 'dropout1_rate': 0.1, 'dense2_units': 16, 'dropout2_rate': 0.1, 'l2_reg': 0.001}
        input_dim = 128

        mlp_builder = MLP(input_shape=input_dim, mlp_params=mlp_params, learning_rate=config_instance.EVAL_LEARNING_RATE)
        model = mlp_builder.build()

        self.assertIsNotNone(model, "MLP model build failed, model is None.")
        self.assertEqual(model.input_shape, (None, input_dim), "MLP input shape mismatch.")
        model.summary(print_fn=lambda x: print(f"  {x}"))
        print("\n  MLPModelBuilder build test passed.")
        print("--- MLPModelBuilder Build Test Complete ---")