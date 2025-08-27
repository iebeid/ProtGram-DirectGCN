# ==============================================================================
# MODULE: testers/models.py
# PURPOSE: Contains tests for model building and instantiation.
# VERSION: 1.0
# AUTHOR: Islam Ebeid
# ==============================================================================

import unittest
import torch
from configuration.config import Config
from source.models.fnn.mlp import MLP
from source.models.factory import ModelFactory
from source.utils.data.data_utils import DataUtils


class ModelBuildTests(unittest.TestCase):
    """A class for testing model building and instantiation."""

    def test_mlp_model_build(self):
        print("\n" + "=" * 80)
        DataUtils.print_header("MLP Model Build Test")
        print("=" * 80)
        config_instance = Config()
        input_dim = 128

        # --- DEFINITIVE FIX: Call the static `build` method correctly ---
        # The previous implementation was trying to instantiate the MLP class, which
        # has no __init__ method and would cause a TypeError. This now correctly
        # calls the static build method, which is the intended use.
        model = MLP.build(input_dim=input_dim, config=config_instance)

        self.assertIsNotNone(model, "MLP model build failed, model is None.")
        self.assertEqual(model.input_shape, (None, input_dim), "MLP input shape mismatch.")
        model.summary(print_fn=lambda x: print(f"  {x}"))
        print("\n  MLPModelBuilder build test passed.")
        print("--- MLPModelBuilder Build Test Complete ---")

    def test_model_factory(self):
        """Tests that the ModelFactory can create various GNN models."""
        print("\n" + "=" * 80)
        DataUtils.print_header("ModelFactory Test")
        print("=" * 80)
        config_instance = Config()
        # Use the 'benchmark' context for testing parameter fetching
        factory = ModelFactory(config_instance, context='benchmark')

        # List of models to test
        models_to_test = ["GCN", "GAT", "GraphSAGE", "DirectGCN"]
        in_channels = 64
        num_classes = 7

        for model_name in models_to_test:
            print(f"  - Testing creation of model: {model_name}")
            model = factory.create_model(
                model_name=model_name,
                in_channels=in_channels,
                num_classes=num_classes
            )
            self.assertIsNotNone(model, f"Factory failed to create model: {model_name}")
            self.assertIsInstance(model, torch.nn.Module, f"{model_name} is not a valid torch.nn.Module")
            print(f"    ✅ Successfully created {model_name}")
        print("\n--- ModelFactory Test Complete ---")