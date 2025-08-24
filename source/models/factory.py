# ==============================================================================
# MODULE: models/factory.py
# PURPOSE: A centralized, dynamic factory for creating all GNN models with caching.
# VERSION: 2.0 (Dynamic registration and instance caching)
# AUTHOR: Gemini Code Assist
# ==============================================================================

import inspect
from typing import Optional, Dict, Any, Callable, Type

import torch.nn as nn

from configuration.config import Config
# --- Import all models that can be registered ---
from source.models.gnn.spectral.chebnet import ChebNet
from source.models.gnn.spectral.directgcn import DirectGCN
from source.models.gnn.spectral.gcn import GCN
from source.models.gnn.spectral.rgcn import RGCN
from source.models.gnn.spectral.dirgnn import DirGNN
from source.models.gnn.spatial.gat import GAT
from source.models.gnn.spatial.gin import GIN
from source.models.gnn.spatial.graphsage import GraphSAGE


class ModelFactory:
    """
    A dynamic, centralized factory for creating GNN models.
    This class uses a decorator-based registration system, making it easy to add
    new models without modifying the factory code. It also includes an in-memory
    cache to avoid re-instantiating identical models during a single run.
    """
    _registry: Dict[str, Type[nn.Module]] = {}
    _instance_cache: Dict[str, nn.Module] = {}

    def __init__(self, config: Config, context: str):
        """
        Initializes the factory with a configuration and a context.

        Args:
            config (Config): The main configuration object.
            context (str): The context of the trainer ('benchmark', 'singleton', 'protgram').
                           This determines which set of config parameters to use.
        """
        self.config = config
        self.context = context
        if self.context not in ['benchmark', 'singleton', 'protgram']:
            raise ValueError(f"Invalid factory context: {self.context}")

    @classmethod
    def register_model(cls, name: str) -> Callable:
        """
        A class method decorator to register a model class with the factory.
        Usage: @ModelFactory.register_model("MyGNN")
        """

        def decorator(model_class: Type[nn.Module]) -> Type[nn.Module]:
            cls._registry[name.lower()] = model_class
            return model_class

        return decorator

    def _get_params(self, model_name_lower: str) -> Dict[str, Any]:
        """
        Gets the appropriate GNN parameters from the config based on the context
        and the specific model being requested.
        """
        # Base parameters applicable to most models
        params = {}
        # --- REFACTOR: Consolidate benchmark and singleton contexts which share a similar structure ---
        if self.context in ['benchmark', 'singleton']:
            prefix = 'BENCHMARK_' if self.context == 'benchmark' else 'SINGLETON_'
            params = {
                'hidden_channels': getattr(self.config, f'{prefix}GNN_HIDDEN_CHANNELS'),
                'num_layers': getattr(self.config, f'{prefix}GNN_NUM_LAYERS'),
                'dropout_rate': getattr(self.config, f'{prefix}GNN_DROPOUT_RATE'),
            }
            if model_name_lower == 'gat':
                params['heads'] = getattr(self.config, f'{prefix}GAT_HEADS')
            elif model_name_lower == 'chebnet':
                params['K'] = getattr(self.config, f'{prefix}CHEBNET_K')
            elif model_name_lower == 'rgcn':
                params['num_relations'] = getattr(self.config, f'{prefix}RGCN_NUM_RELATIONS')
            elif model_name_lower == 'directgcn':
                params['layer_dims_config'] = getattr(self.config, f'{prefix}DIRECTGCN_HIDDEN_LAYER_DIMS')

        elif self.context == 'protgram':
            params = {
                'hidden_channels': self.config.PROTGRAM_GNN_HIDDEN_CHANNELS,
                'num_layers': self.config.PROTGRAM_GNN_NUM_LAYERS,
                'dropout_rate': self.config.PROTGRAM_DROPOUT_RATE,
            }
            if model_name_lower == 'rgcn':
                params['num_relations'] = 2  # Default for protgram context
            elif model_name_lower == 'directgcn':
                params['layer_dims_config'] = self.config.DIRECTGCN_HIDDEN_LAYER_DIMS

        # Add common DirectGCN parameters if it's the requested model
        if model_name_lower == 'directgcn':
            params.update({
                'one_gram_dim': self.config.PROTGRAM_1GRAM_INIT_DIM,
                'max_pe_len': self.config.PROTGRAM_MAX_PE_LEN,
                'dropout': self.config.PROTGRAM_DROPOUT_RATE,
                'gating_mode': self.config.PROTGRAM_GATING_COEFF_MODE,
                'disable_pe': not self.config.PROTGRAM_USE_POSITIONAL_EMBEDDING
            })
        return params
        # --- DEFINITIVE FIX: Ensure use_homo_hetero_paths is always present for DirectGCN ---
        if model_name_lower == 'directgcn':
            params.setdefault('use_homo_hetero_paths', False)
        return params

    def create_model(self, model_name: str, in_channels: int, num_classes: int, **kwargs) -> Optional[nn.Module]:
        """
        Creates and returns a GNN model instance using the registry and cache.
        """
        name_lower = model_name.lower()
        model_class = self._registry.get(name_lower)

        if not model_class:
            raise ValueError(f"Unknown model name: '{model_name}'. Supported models are: {list(self._registry.keys())}")

        # --- NEW: In-memory instance caching ---
        # Create a unique key based on model name, context, and key parameters
        cache_key_parts = [name_lower, self.context, in_channels, num_classes]
        # Add other relevant kwargs to the key to ensure uniqueness
        for k, v in sorted(kwargs.items()):
            cache_key_parts.append(f"{k}={v}")
        cache_key = ":".join(map(str, cache_key_parts))

        if cache_key in self._instance_cache:
            print(f"  Reusing cached instance of '{model_name}' for context '{self.context}'.")
            return self._instance_cache[cache_key]

        print(f"  Creating new instance of '{model_name}' for context '{self.context}'.")
        # --- Get context-specific and model-specific parameters ---
        params = self._get_params(name_lower)

        # --- Build the final argument dictionary for the model's constructor ---
        constructor_args = {
            'in_channels': in_channels,
            'out_channels': num_classes,
        }
        constructor_args.update(params)
        constructor_args.update(kwargs)

        # Special handling for DirectGCN's complex parameters
        if name_lower == 'directgcn':
            layer_dims_config = constructor_args.pop('layer_dims_config', [])
            constructor_args['layer_dims'] = [in_channels] + layer_dims_config
            graph_obj = constructor_args.get('graph_obj')
            constructor_args['num_graph_nodes'] = getattr(graph_obj, 'number_of_nodes', getattr(graph_obj, 'num_nodes', 0))
            constructor_args['task_num_output_classes'] = num_classes
            constructor_args['n_gram_len'] = kwargs.get('n_val', 1)

        # --- Filter args to only those the constructor accepts ---
        model_signature = inspect.signature(model_class.__init__)
        valid_args = {k: v for k, v in constructor_args.items() if k in model_signature.parameters}

        try:
            model_instance = model_class(**valid_args)
            self._instance_cache[cache_key] = model_instance
            return model_instance
        except TypeError as e:
            print(f"  ERROR: Failed to instantiate model '{model_name}'. Mismatch between provided and expected arguments.")
            print(f"    Provided: {sorted(constructor_args.keys())}")
            print(f"    Expected: {sorted(model_signature.parameters.keys())}")
            print(f"    Error: {e}")
            return None


# --- Register all known models ---
# This makes the factory aware of them without needing a large if/elif block.
# In a larger project, the @ModelFactory.register_model decorator could be
# placed directly on the class definition in the model's own file.
ModelFactory.register_model("gcn")(GCN)
ModelFactory.register_model("graphsage")(GraphSAGE)
ModelFactory.register_model("gin")(GIN)
ModelFactory.register_model("dirgnn")(DirGNN)
ModelFactory.register_model("gat")(GAT)
ModelFactory.register_model("chebnet")(ChebNet)
ModelFactory.register_model("rgcn")(RGCN)
ModelFactory.register_model("directgcn")(DirectGCN)