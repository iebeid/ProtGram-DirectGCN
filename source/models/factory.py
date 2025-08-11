# ==============================================================================
# MODULE: models/factory.py
# PURPOSE: A centralized factory for creating all GNN models.
# VERSION: 1.0
# AUTHOR: Gemini Code Assist
# ==============================================================================

from typing import Optional, Dict, Any

import torch.nn as nn

from configuration.config import Config
from source.data_builders.graph import DirectedNgramGraph
from source.models.gnn.spectral.chebnet import ChebNet
from source.models.gnn.spectral.directgcn import DirectGCN
from source.models.gnn.spectral.gcn import GCN
from source.models.gnn.spectral.rgcn import RGCN
from source.models.gnn.spectral.tongidigcn import TongDiGCN
from source.models.gnn.spatial.gat import GAT
from source.models.gnn.spatial.gin import GIN
from source.models.gnn.spatial.graphsage import GraphSAGE


class ModelFactory:
    """
    A centralized factory for creating GNN models.
    This class eliminates code duplication across different trainers by providing a
    single, context-aware point of model instantiation.
    """

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

    def _get_params(self) -> Dict[str, Any]:
        """Gets the appropriate GNN parameters from the config based on the context."""
        if self.context == 'benchmark':
            return {
                'hidden_channels': self.config.BENCHMARK_GNN_HIDDEN_CHANNELS,
                'num_layers': self.config.BENCHMARK_GNN_NUM_LAYERS,
                'dropout_rate': self.config.BENCHMARK_GNN_DROPOUT_RATE,
                'gat_heads': self.config.BENCHMARK_GAT_HEADS,
                'chebnet_k': self.config.BENCHMARK_CHEBNET_K,
                'rgcn_num_relations': self.config.BENCHMARK_RGCN_NUM_RELATIONS
            }
        elif self.context == 'singleton':
            return {
                'hidden_channels': self.config.SINGLETON_GNN_HIDDEN_CHANNELS,
                'num_layers': self.config.SINGLETON_GNN_NUM_LAYERS,
                'dropout_rate': self.config.SINGLETON_GNN_DROPOUT_RATE,
                'gat_heads': self.config.SINGLETON_GAT_HEADS,
                'chebnet_k': self.config.SINGLETON_CHEBNET_K,
                'rgcn_num_relations': self.config.SINGLETON_RGCN_NUM_RELATIONS
            }
        elif self.context == 'protgram':
            return {
                'hidden_channels': self.config.PROTGRAM_GNN_HIDDEN_CHANNELS,
                'num_layers': self.config.PROTGRAM_GNN_NUM_LAYERS,
                'dropout_rate': self.config.PROTGRAM_DROPOUT_RATE,
                'rgcn_num_relations': 2  # Default for protgram context
            }
        return {}

    def create_model(self, model_name: str, in_channels: int, num_classes: int, **kwargs) -> Optional[nn.Module]:
        """
        Creates and returns a GNN model instance.

        Args:
            model_name (str): The name of the model to create (e.g., "GCN", "DirectGCN").
            in_channels (int): The number of input features for the model.
            num_classes (int): The number of output classes for the model.
            **kwargs: Additional model-specific arguments (e.g., `graph_obj` for DirectGCN).

        Returns:
            An instantiated PyTorch GNN model, or None if the model name is unknown.
        """
        params = self._get_params()
        model_params = {'in_channels': in_channels, 'hidden_channels': params['hidden_channels'],
                        'out_channels': num_classes, 'num_layers': params['num_layers'],
                        'dropout_rate': params['dropout_rate']}

        name_lower = model_name.lower()
        if name_lower == 'gcn': return GCN(**model_params)
        if name_lower == 'graphsage': return GraphSAGE(**model_params)
        if name_lower == 'gin': return GIN(**model_params)
        if name_lower == 'tongdigcn': return TongDiGCN(**model_params)
        if name_lower == 'gat':
            return GAT(**model_params, heads=params['gat_heads'])
        if name_lower == 'chebnet':
            return ChebNet(**model_params, K=params['chebnet_k'])
        if name_lower == 'rgcn':
            return RGCN(**model_params, num_relations=params['rgcn_num_relations'])
        if name_lower == 'directgcn':
            # --- FIX: Use the correct, context-specific layer dimensions for DirectGCN ---
            if self.context == 'singleton':
                layer_dims_config = self.config.SINGLETON_DIRECTGCN_HIDDEN_LAYER_DIMS
            elif self.context == 'benchmark':
                layer_dims_config = self.config.BENCHMARK_DIRECTGCN_HIDDEN_LAYER_DIMS
            else:  # protgram
                layer_dims_config = self.config.DIRECTGCN_HIDDEN_LAYER_DIMS

            # --- FIX: Handle both PyG Data (num_nodes) and custom Graph (number_of_nodes) objects ---
            graph_obj = kwargs.get('graph_obj')
            num_nodes = getattr(graph_obj, 'number_of_nodes', getattr(graph_obj, 'num_nodes', 0))

            layer_dims = [in_channels] + layer_dims_config
            return DirectGCN(layer_dims=layer_dims, num_graph_nodes=num_nodes,
                             task_num_output_classes=num_classes, n_gram_len=kwargs.get('n_val', 1),
                             use_homo_hetero_paths=kwargs.get('use_homo_hetero_paths', False),
                             one_gram_dim=self.config.PROTGRAM_1GRAM_INIT_DIM, max_pe_len=self.config.PROTGRAM_MAX_PE_LEN,
                             dropout=self.config.PROTGRAM_DROPOUT_RATE, gating_mode=self.config.PROTGRAM_GATING_COEFF_MODE)

        print(f"  ERROR: Unknown model type '{model_name}' requested from factory.")
        return None