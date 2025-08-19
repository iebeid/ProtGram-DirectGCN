# ==============================================================================
# MODULE: data_builders/graph.py
# PURPOSE: Contains robust classes for n-gram graph representation with optimized I/O.
# VERSION: 12.0 (Added performant serialization/deserialization methods)
# AUTHOR: Islam Ebeid (Refactored by Gemini Code Assist)
# ==============================================================================
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
import torch


class Graph:
    """A base class for representing n-gram graphs with nodes and edges."""

    def __init__(self, nodes: Dict[int, Any], edges: List[Tuple]):  # nodes keys are int IDs
        self.idx_to_node_map_from_constructor = nodes if nodes is not None else {}
        self.original_edges = edges if edges is not None else []

        self.node_to_idx: Dict[Any, int] = {}
        self.idx_to_node: Dict[int, Any] = {}
        self.number_of_nodes: int = 0
        self.node_names: List[Any] = []
        self.edges: List[Tuple] = []
        self.number_of_edges: int = 0

        self._process_constructor_inputs()

    def _process_constructor_inputs(self):
        """
        Processes the nodes and edges passed to the constructor.
        Assumes `nodes` is a map from integer index to node name (e.g., n-gram string).
        Assumes `edges` contains tuples where the first two elements are integer indices.
        """
        if not self.idx_to_node_map_from_constructor and not self.original_edges:
            self.number_of_nodes = 0
            self.edges = []
            self.number_of_edges = 0
            return

        all_integer_indices = set()
        if self.idx_to_node_map_from_constructor:
            all_integer_indices.update(self.idx_to_node_map_from_constructor.keys())

        for edge_tuple in self.original_edges:
            if len(edge_tuple) >= 2:
                if not isinstance(edge_tuple[0], (int, np.integer)) or \
                        not isinstance(edge_tuple[1], (int, np.integer)):
                    continue
                all_integer_indices.add(int(edge_tuple[0]))
                all_integer_indices.add(int(edge_tuple[1]))

        if not all_integer_indices and not self.idx_to_node_map_from_constructor:
            self.number_of_nodes = 0
            return

        max_node_map_idx = -1
        if self.idx_to_node_map_from_constructor:
            valid_node_indices = {idx for idx in self.idx_to_node_map_from_constructor.keys() if
                                  isinstance(idx, (int, np.integer)) and idx >= 0}
            # --- FIX: Add a warning for potential data inconsistency ---
            if all_integer_indices and valid_node_indices and max(all_integer_indices) > max(valid_node_indices):
                # --- ENHANCEMENT: Make the warning more specific and explain the consequence ---
                print(f"  - WARNING: Data inconsistency detected. Max edge index ({max(all_integer_indices)}) "
                      f"exceeds max node map index ({max(valid_node_indices)}). This can lead to silent data loss.")
            if valid_node_indices:
                max_node_map_idx = max(valid_node_indices)

        max_edge_idx = -1
        if all_integer_indices:
            max_edge_idx = max(all_integer_indices)

        self.number_of_nodes = max(max_node_map_idx, max_edge_idx) + 1

        temp_idx_to_node_name = {}
        for i in range(self.number_of_nodes):
            node_name = self.idx_to_node_map_from_constructor.get(i)
            if node_name is None:
                node_name = f"__NODE_{i}__"
            temp_idx_to_node_name[i] = str(node_name)

        self.idx_to_node = temp_idx_to_node_name
        self.node_to_idx = {name: idx for idx, name in self.idx_to_node.items()}
        self.node_names = [self.idx_to_node.get(i, f"__NODE_{i}__") for i in range(self.number_of_nodes)]

        self.edges = self.original_edges
        self.number_of_edges = len(self.edges)

    def get_node_to_idx_map(self) -> Dict[str, int]:
        """Returns a copy of the node name to index mapping."""
        return self.node_to_idx.copy()

    @staticmethod
    def _sparse_identity(size: int, device: torch.device) -> torch.Tensor:
        """Creates a sparse identity matrix of given size."""
        if size <= 0:
            empty_indices = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_values = torch.empty(0, dtype=torch.float32, device=device)
            valid_size = max(0, size)
            return torch.sparse_coo_tensor(empty_indices, empty_values, (valid_size, valid_size)).coalesce()

        indices = torch.arange(size, device=device).unsqueeze(0).repeat(2, 1)
        values = torch.ones(size, device=device, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()

    def save_to_dir(self, dir_path: Union[str, Path]):
        """
        Saves the graph object's components to a directory for robust,
        performant serialization, avoiding pickle.
        """
        from source.utils.fs.file_utils import FileUtils
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            'number_of_nodes': self.number_of_nodes,
            'number_of_edges': self.number_of_edges,
            'n_value': getattr(self, 'n_value', None),
            'edge_file_path': str(getattr(self, 'edge_file_path', None)),
            'epsilon_propagation': getattr(self, 'epsilon_propagation', None)
        }
        FileUtils.save_json(metadata, dir_path / "metadata.json")

        # Save node map
        node_df = pd.DataFrame(self.idx_to_node.items(), columns=['id', 'node_name'])
        node_df.to_parquet(dir_path / "nodes.parquet", index=False)

        # --- NEW: Save sparse tensor attributes ---
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor) and attr_value.is_sparse:
                coalesced_tensor = attr_value.coalesce()
                indices = coalesced_tensor.indices().cpu().numpy()
                values = coalesced_tensor.values().cpu().numpy()
                np.save(dir_path / f"{attr_name}_indices.npy", indices)
                np.save(dir_path / f"{attr_name}_values.npy", values)
                # Add shape to metadata
                metadata[f"{attr_name}_shape"] = list(coalesced_tensor.shape)

        # Re-save metadata with sparse tensor shapes
        FileUtils.save_json(metadata, dir_path / "metadata.json")

        print(f"  Graph components saved to directory: {dir_path}")

    @classmethod
    def load_from_dir(cls, dir_path: Union[str, Path]) -> 'DirectedNgramGraph':
        """
        Loads a graph object by reconstructing it from its saved components,
        avoiding pickle.
        """
        from source.utils.fs.file_utils import FileUtils
        from source.data_structures.direct_ngram_graph import DirectedNgramGraph

        dir_path = Path(dir_path)
        metadata_path = dir_path / "metadata.json"
        nodes_path = dir_path / "nodes.parquet"

        if not all([metadata_path.exists(), nodes_path.exists()]):
            raise FileNotFoundError(f"Cannot load graph from '{dir_path}', component files are missing.")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        nodes_df = pd.read_parquet(nodes_path)
        idx_to_node = dict(zip(nodes_df['id'], nodes_df['node_name']))

        # Re-instantiate the class using the loaded components
        # --- FIX: Pass the directory path to the constructor for robust loading ---
        metadata['dir_path'] = str(dir_path)
        return DirectedNgramGraph(nodes=idx_to_node, **metadata)
