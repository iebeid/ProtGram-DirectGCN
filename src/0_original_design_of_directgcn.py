import sys
import time
import re
import random
import pickle
import gc
import tracemalloc
import traceback
from operator import itemgetter
from copy import deepcopy
from contextlib import redirect_stdout

from collections import defaultdict, Counter, deque
from itertools import chain
from pathlib import Path

import chardet
# from tqdm import tqdm # Not explicitly used in this version
import numpy as np
import scipy as sp
from scipy.spatial import distance as sp_distance
from scipy import stats as sp_stats
import pandas as pd
import networkx as nx

import sklearn as skl
from sklearn.manifold import TSNE
import tensorflow as tf

from gensim.models import Word2Vec
from Levenshtein import distance as levenshtein_distance, ratio as levenshtein_ratio

import matplotlib.pyplot as plt
import seaborn as sns

# --- PyTorch Geometric Imports ---
try:
    import torch
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import to_networkx  # For converting PyG data to NetworkX

    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: PyTorch or PyTorch Geometric not found. Planetoid dataset loading will not work.")
    print(
        "Please install PyTorch (https://pytorch.org/) and PyTorch Geometric (https://pytorch-geometric.readthedocs.io/).")
# --- End PyTorch Geometric Imports ---

random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)
tf.config.run_functions_eagerly(False)
tf.config.threading.set_inter_op_parallelism_threads(8)


# --- Utility Classes ---
class Shape:
    def __init__(self, in_size, out_size, batch_size):
        self.in_size = in_size
        self.out_size = out_size
        self.batch_size = batch_size

    def __str__(self): return f"(In: {self.in_size}, Out: {self.out_size}, Batch: {self.batch_size})"


class InputOp:
    def __init__(self, input_tensor, normalize):
        self.input_tensor = input_tensor
        if normalize:
            norms = tf.norm(self.input_tensor, axis=1, keepdims=True)
            self.output = self.input_tensor / (norms + 1e-9)
        else:
            self.output = self.input_tensor


class Hyperparameters(dict):
    def add(self, key: str, value):
        self[key] = value

    def get(self, key: str, default=None):
        if key not in self:
            ratios = {levenshtein_ratio(key, k): k for k in self.keys()}
            if ratios and max(ratios.keys()) > 0.7: return self[ratios[max(ratios.keys())]]
            if default is not None: return default
            raise KeyError(f"Hyperparameter '{key}' not found.")
        return super().get(key, default)


class MachineLearning: pass  # Base for organization


class Initializers(MachineLearning):
    @staticmethod
    def _get_init_shape(shape_obj: Shape, axis=1):
        if axis == 1: return (1, shape_obj.out_size)
        return (shape_obj.in_size, shape_obj.out_size)

    @staticmethod
    def get(initializer_name: str, shape_obj: Shape, axis=1, **kwargs):
        keras_initializers = {
            "identity": tf.keras.initializers.Identity(), "uniform": tf.keras.initializers.RandomUniform(0, 1),
            "glorot": tf.keras.initializers.GlorotUniform(), "zeros": tf.keras.initializers.Zeros(),
            "ones": tf.keras.initializers.Ones(),
        }
        key = initializer_name.lower()
        if key not in keras_initializers: raise ValueError(f"Unknown initializer: {initializer_name}")
        initializer = keras_initializers[key]
        if kwargs:
            try:
                config = initializer.get_config()
                config.update(kwargs)
                initializer = type(initializer).from_config(config)
            except Exception as e:
                print(f"Warn: Could not apply kwargs to {initializer_name}: {e}")
        return initializer(shape=Initializers._get_init_shape(shape_obj, axis), dtype=tf.float32)


class LossFunctions(MachineLearning):
    @staticmethod
    def masked_cross_entropy_loss_evaluater_2(predictions, y, mask):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y)
        mask_float = tf.cast(mask, dtype=tf.float32)
        mask_sum = tf.reduce_sum(mask_float)
        if mask_sum > 0:
            # Normalize loss contribution by the number of true elements in the mask
            loss = tf.boolean_mask(loss, mask)  # Apply mask to loss
            return tf.reduce_mean(loss)  # Mean of losses where mask is true
        else:
            return tf.constant(0.0, dtype=loss.dtype)


class Evaluation(MachineLearning):
    @staticmethod
    def masked_accuracy_evaluater(prediction, y, mask):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        masked_accuracy_values = tf.boolean_mask(accuracy_all, mask)
        if tf.size(masked_accuracy_values) == 0:
            return tf.constant(0.0, dtype=tf.float32)
        return tf.reduce_mean(masked_accuracy_values)


class Optimizers(MachineLearning):
    @staticmethod
    def optimizer(learn_rate):
        return tf.keras.optimizers.Adam(learning_rate=learn_rate)


class Algorithms:  # Simplified, assuming not used or defined elsewhere if needed by user
    pass


# --- Graph Classes ---
class Graph:
    def __init__(self, nodes=None, edges=None):
        self.original_nodes_dict = nodes if nodes is not None else {}
        self.original_edges = edges if edges is not None else []
        self.nodes = {}
        self.edges = []
        self.node_index = {}
        self.node_inverted_index = {}
        self.number_of_nodes = 0
        self.number_of_edges = 0
        self._features = None
        self.dimensions = 0
        self.y_labels_one_hot = None
        self.class_to_int_mapping = {}
        self.int_to_class_mapping = {}
        self.node_label_profile = defaultdict(list)
        self.number_of_classes = 0
        self.node_indices()
        self.edge_indices()

    def node_indices(self):
        current_nodes_orig_id_map = self.original_nodes_dict
        if not current_nodes_orig_id_map and self.original_edges:
            all_node_ids_from_edges = set()
            for pair_idx in range(len(self.original_edges)):
                if len(self.original_edges[pair_idx]) >= 2:
                    all_node_ids_from_edges.add(str(self.original_edges[pair_idx][0]))
                    all_node_ids_from_edges.add(str(self.original_edges[pair_idx][1]))
            current_nodes_orig_id_map = {node_id: "1" for node_id in sorted(list(all_node_ids_from_edges))}
            self.original_nodes_dict = current_nodes_orig_id_map

        sorted_original_node_ids = sorted(list(current_nodes_orig_id_map.keys()))
        new_nodes_indexed_map = {}
        self.node_index = {}
        self.node_inverted_index = {}
        for i, orig_node_id_str_key in enumerate(sorted_original_node_ids):
            orig_node_id_str = str(orig_node_id_str_key)
            self.node_index[orig_node_id_str] = i
            self.node_inverted_index[i] = orig_node_id_str
            new_nodes_indexed_map[i] = current_nodes_orig_id_map[orig_node_id_str_key]
        self.nodes = new_nodes_indexed_map
        self.number_of_nodes = len(self.nodes)

    def edge_indices(self):
        if not self.node_index:
            print(f"Warning ({type(self).__name__}): Node indices not created. Cannot index edges.")
            self.edges = deepcopy(self.original_edges)
            self.number_of_edges = len(self.original_edges)
            return
        indexed_edge_list = []
        for edge_data in self.original_edges:
            if len(edge_data) < 2:
                print(f"Warning ({type(self).__name__}): Edge data {edge_data} too short. Skipping.")
                continue
            try:
                source_node_original_id_str = str(edge_data[0])
                target_node_original_id_str = str(edge_data[1])
                idx_s = self.node_index[source_node_original_id_str]
                idx_t = self.node_index[target_node_original_id_str]
                indexed_edge_list.append((idx_s, idx_t) + tuple(edge_data[2:]))
            except KeyError as e:
                print(
                    f"Warning ({type(self).__name__}): Node ID {e} from edge {edge_data} not in node_index. Skipping edge.")
            except IndexError:
                print(f"Warning ({type(self).__name__}): Edge data {edge_data} has unexpected format. Skipping.")
        self.edges = indexed_edge_list
        self.number_of_edges = len(self.edges)

    def set_node_features(self, features_data_np, default_dim_if_none=None):
        if features_data_np is not None and isinstance(features_data_np, np.ndarray):
            if features_data_np.shape[0] == self.number_of_nodes or (
                    self.number_of_nodes == 0 and features_data_np.shape[0] == 0):
                self._features = tf.constant(features_data_np, dtype=tf.float32)
                self.dimensions = features_data_np.shape[1] if features_data_np.ndim > 1 and features_data_np.shape[
                    0] > 0 else 0
            else:
                print(
                    f"Warning ({type(self).__name__}): Feature array shape {features_data_np.shape} incompatible with num_nodes {self.number_of_nodes}.")
                self._features = None;
                self.dimensions = 0
        elif default_dim_if_none is not None and self.number_of_nodes > 0:
            self._features = tf.eye(self.number_of_nodes, num_columns=default_dim_if_none,
                                    dtype=tf.float32) if default_dim_if_none > 0 else tf.zeros(
                (self.number_of_nodes, 0), dtype=tf.float32)
            self.dimensions = default_dim_if_none
        else:
            self._features = None;
            self.dimensions = 0
            if self.number_of_nodes > 0: print(
                f"Warning ({type(self).__name__}): No features provided for {self.number_of_nodes} nodes.")

    @property
    def features(self):
        if self._features is None:
            if self.number_of_nodes > 0:
                hp_input_dim = self.hp.get('input_layer_dimension', 32) if hasattr(self, 'hp') else 32
                self.set_node_features(None, default_dim_if_none=hp_input_dim)
            else:
                self.set_node_features(np.array([]).reshape(0, 0))
        return self._features

    def set_node_labels_from_graph_nodes(self, dummy_community_detection=False):
        self.node_label_profile = defaultdict(list)
        self.number_of_classes = 0
        y_list = []
        if not dummy_community_detection:
            if not self.nodes: self.y_labels_one_hot = tf.zeros((self.number_of_nodes, 0), dtype=tf.float32); return
            unique_labels = sorted(list(set(str(v) for v in self.nodes.values() if v not in ["N/A", None])))
            self.class_to_int_mapping = {lbl: i for i, lbl in enumerate(unique_labels)}
            self.int_to_class_mapping = {i: lbl for lbl, i in self.class_to_int_mapping.items()}
            self.number_of_classes = len(unique_labels)
            if self.number_of_nodes > 0 and self.number_of_classes > 0:
                for i in range(self.number_of_nodes):
                    lbl_val = self.nodes.get(i)
                    y_vec = np.zeros(self.number_of_classes, dtype=np.float32)
                    if lbl_val not in ["N/A", None]:
                        cls_int = self.class_to_int_mapping.get(str(lbl_val))
                        if cls_int is not None: y_vec[cls_int] = 1.0; self.node_label_profile[str(lbl_val)].append(i)
                    y_list.append(y_vec)
                self.y_labels_one_hot = tf.constant(y_list, dtype=tf.float32)
            else:
                self.y_labels_one_hot = tf.zeros(
                    (self.number_of_nodes, self.number_of_classes if self.number_of_classes > 0 else 0),
                    dtype=tf.float32)
            self.node_label_profile = dict(self.node_label_profile)
        else:
            if not self.edges or self.number_of_nodes == 0: self.y_labels_one_hot = tf.zeros((self.number_of_nodes, 0),
                                                                                             dtype=tf.float32); return
            nx_g = nx.Graph([(e[0], e[1]) for e in self.edges])
            try:
                louvain_sets = nx.community.louvain_communities(nx_g, seed=123)
            except Exception as e:
                print(f"Louvain failed: {e}");
                self.y_labels_one_hot = tf.zeros((self.number_of_nodes, 0), dtype=tf.float32);
                return
            if not louvain_sets: self.y_labels_one_hot = tf.zeros((self.number_of_nodes, 0), dtype=tf.float32); return
            self.number_of_classes = len(louvain_sets)
            node_to_cid = {}
            for cid, comm in enumerate(louvain_sets):
                for node_idx in comm:
                    if node_idx < self.number_of_nodes: node_to_cid[node_idx] = cid; self.node_label_profile[
                        str(cid)].append(node_idx)
            if self.number_of_nodes > 0 and self.number_of_classes > 0:
                y_l = [np.zeros(self.number_of_classes, dtype=np.float32) for _ in range(self.number_of_nodes)]
                for i in range(self.number_of_nodes):
                    cid = node_to_cid.get(i)
                    if cid is not None: y_l[i][cid] = 1.0
                self.y_labels_one_hot = tf.constant(y_l, dtype=tf.float32)
            else:
                self.y_labels_one_hot = tf.zeros(
                    (self.number_of_nodes, self.number_of_classes if self.number_of_classes > 0 else 0),
                    dtype=tf.float32)
            self.node_label_profile = dict(self.node_label_profile)

    def node_labels_sampler(self, split_percent):
        if self.number_of_nodes == 0 or not self.node_label_profile or self.number_of_classes == 0:
            empty_mask = np.zeros(self.number_of_nodes, dtype=bool)
            print(f"Warning ({type(self).__name__}): Cannot sample labels.")
            return empty_mask, empty_mask, empty_mask
        all_node_indices = list(range(self.number_of_nodes))
        k_train_total = np.floor((split_percent * self.number_of_nodes) / 100)
        final_train_samples_indices = []
        samples_per_class_train = np.floor(
            k_train_total / self.number_of_classes) if self.number_of_classes > 0 else k_train_total
        for _, node_indices_in_class in self.node_label_profile.items():
            num_to_sample = min(len(node_indices_in_class),
                                int(samples_per_class_train if samples_per_class_train > 0 else (
                                    1 if k_train_total > 0 and len(node_indices_in_class) > 0 else 0)))
            if num_to_sample > 0: final_train_samples_indices.extend(list(
                np.random.choice(np.array(node_indices_in_class, dtype=np.int32), size=num_to_sample, replace=False)))
        if not final_train_samples_indices and k_train_total > 0 and all_node_indices: final_train_samples_indices = list(
            np.random.choice(all_node_indices, size=min(int(k_train_total), len(all_node_indices)), replace=False))
        final_train_samples_indices = sorted(list(set(final_train_samples_indices)))
        rest_samples_indices = sorted(list(set(all_node_indices) - set(final_train_samples_indices)))
        np.random.shuffle(rest_samples_indices)
        middle_index = len(rest_samples_indices) // 2
        valid_samples_indices = rest_samples_indices[:middle_index]
        test_samples_indices = rest_samples_indices[middle_index:]
        train_mask = np.zeros(self.number_of_nodes, dtype=bool)
        valid_mask = np.zeros(self.number_of_nodes, dtype=bool)
        test_mask = np.zeros(self.number_of_nodes, dtype=bool)
        if final_train_samples_indices: train_mask[final_train_samples_indices] = True
        if valid_samples_indices: valid_mask[valid_samples_indices] = True
        if test_samples_indices: test_mask[test_samples_indices] = True
        return train_mask, test_mask, valid_mask


class UndirectedGraph(Graph):
    def __init__(self, edges=None, nodes=None):
        super().__init__(nodes=nodes, edges=edges)
        self._adjacency = None;
        self._weighted_adjacency = None
        self._degree_matrix = None
        self._weighted_degree_matrix = None
        self._degree_normalized_adjacency = None

    def __str__(self):
        return f"An undirected graph of {self.number_of_nodes} nodes and {self.number_of_edges} edges."

    @property
    def adjacency(self):
        if self._adjacency is None:
            if self.number_of_nodes == 0: self._adjacency = np.array([], dtype=float).reshape(0,
                                                                                              0); return self._adjacency
            adj_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
            for s_idx, t_idx, *_ in self.edges: adj_matrix[s_idx, t_idx] = 1.0; adj_matrix[t_idx, s_idx] = 1.0
            self._adjacency = adj_matrix
        return self._adjacency

    @property
    def weighted_adjacency(self):
        if self._weighted_adjacency is None:
            if self.number_of_nodes == 0: self._weighted_adjacency = np.array([], dtype=float).reshape(0,
                                                                                                       0); return self._weighted_adjacency
            adj_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
            for s_idx, t_idx, weight, *_ in self.edges: adj_matrix[s_idx, t_idx] = float(weight); adj_matrix[
                t_idx, s_idx] = float(weight)
            self._weighted_adjacency = adj_matrix
        return self._weighted_adjacency

    @property
    def degree_matrix(self):
        if self._degree_matrix is None:
            if self.number_of_nodes == 0: self._degree_matrix = np.array([], dtype=float).reshape(0,
                                                                                                  0); return self._degree_matrix
            self._degree_matrix = np.diag(np.sum(self.adjacency, axis=0))
        return self._degree_matrix

    @property
    def weighted_degree_matrix(self):
        if self._weighted_degree_matrix is None:
            if self.number_of_nodes == 0: self._weighted_degree_matrix = np.array([], dtype=float).reshape(0,
                                                                                                           0); return self._weighted_degree_matrix
            self._weighted_degree_matrix = np.diag(np.sum(self.weighted_adjacency, axis=0))
        return self._weighted_degree_matrix

    @property
    def degree_normalized_adjacency(self):
        if self._degree_normalized_adjacency is None:
            if self.number_of_nodes == 0: self._degree_normalized_adjacency = np.array([], dtype=float).reshape(0,
                                                                                                                0); return self._degree_normalized_adjacency
            adj_tilde = self.adjacency + np.identity(self.number_of_nodes, dtype=float)
            degree_tilde_vec = np.sum(adj_tilde, axis=0)
            inv_sqrt_degree_tilde = np.power(degree_tilde_vec, -0.5, where=degree_tilde_vec != 0,
                                             out=np.zeros_like(degree_tilde_vec))
            D_inv_sqrt = np.diag(inv_sqrt_degree_tilde)
            self._degree_normalized_adjacency = D_inv_sqrt @ adj_tilde @ D_inv_sqrt
        return self._degree_normalized_adjacency


class DirectedGraph(Graph):
    def __init__(self, edges=None, nodes=None):
        super().__init__(nodes=nodes, edges=edges)
        self._out_adjacency = None
        self._in_adjacency = None
        self._out_weighted_adjacency = None
        self._in_weighted_adjacency = None
        self._out_degree_matrix = None
        self._in_degree_matrix = None
        self._out_weighted_degree_matrix = None
        self._in_weighted_degree_matrix = None
        self._normalized_weighted_out_adjacecny = None
        self._normalized_weighted_in_adjacecny = None
        self._degree_normalized_weighted_in_adj_symm = None
        self._degree_normalized_weighted_out_adj_symm = None

    def __str__(self):
        return f"A directed graph of {self.number_of_nodes} nodes and {self.number_of_edges} edges."

    def _calculate_adjacencies(self):
        if self._out_adjacency is None and self.number_of_nodes > 0:
            self._out_adjacency = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
            self._in_adjacency = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
            self._out_weighted_adjacency = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
            self._in_weighted_adjacency = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
            for s, t, w, *_ in self.edges: self._out_adjacency[s, t] = 1.; self._in_adjacency[t, s] = 1.;
            self._out_weighted_adjacency[s, t] = float(w); self._in_weighted_adjacency[t, s] = float(w)
        elif self.number_of_nodes == 0:
            [setattr(self, a, np.array([], dtype=float).reshape(0, 0)) for a in
             ['_out_adjacency', '_in_adjacency', '_out_weighted_adjacency', '_in_weighted_adjacency']]

    @property
    def out_adjacency(self):
        if self._out_adjacency is None: self._calculate_adjacencies()
        return self._out_adjacency

    @property
    def in_adjacency(self):
        if self._in_adjacency is None: self._calculate_adjacencies()
        return self._in_adjacency

    @property
    def out_weighted_adjacency(self):
        if self._out_weighted_adjacency is None: self._calculate_adjacencies()
        return self._out_weighted_adjacency

    @property
    def in_weighted_adjacency(self):
        if self._in_weighted_adjacency is None: self._calculate_adjacencies()
        return self._in_weighted_adjacency

    def _calculate_degree_matrices(self):
        if self._out_degree_matrix is None and self.number_of_nodes > 0:
            self._out_degree_matrix = np.diag(np.sum(self.out_adjacency, axis=1));
            self._in_degree_matrix = np.diag(np.sum(self.in_adjacency, axis=1))
            self._out_weighted_degree_matrix = np.diag(np.sum(self.out_weighted_adjacency, axis=1));
            self._in_weighted_degree_matrix = np.diag(np.sum(self.in_weighted_adjacency, axis=1))
        elif self.number_of_nodes == 0:
            [setattr(self, a, np.array([], dtype=float).reshape(0, 0)) for a in
             ['_out_degree_matrix', '_in_degree_matrix', '_out_weighted_degree_matrix', '_in_weighted_degree_matrix']]

    @property
    def out_degree_matrix(self):
        if self._out_degree_matrix is None: self._calculate_degree_matrices()
        return self._out_degree_matrix

    @property
    def in_degree_matrix(self):
        if self._in_degree_matrix is None: self._calculate_degree_matrices()
        return self._in_degree_matrix

    @property
    def out_weighted_degree_matrix(self):
        if self._out_weighted_degree_matrix is None: self._calculate_degree_matrices()
        return self._out_weighted_degree_matrix

    @property
    def in_weighted_degree_matrix(self):
        if self._in_weighted_degree_matrix is None: self._calculate_degree_matrices()
        return self._in_weighted_degree_matrix

    @property
    def degree_normalized_weighted_out_adjacency(self):
        if self._normalized_weighted_out_adjacecny is None:
            if self.number_of_nodes == 0: self._normalized_weighted_out_adjacecny = np.array([], dtype=float).reshape(0,
                                                                                                                      0); return self._normalized_weighted_out_adjacecny
            D_diag = np.diag(self.out_weighted_degree_matrix)
            D_inv_diag = np.power(D_diag, -1, where=D_diag != 0, out=np.zeros_like(D_diag))
            self._normalized_weighted_out_adjacecny = np.diag(D_inv_diag) @ self.out_weighted_adjacency
        return self._normalized_weighted_out_adjacecny

    @property
    def degree_normalized_weighted_in_adjacency(self):
        if self._normalized_weighted_in_adjacecny is None:
            if self.number_of_nodes == 0: self._normalized_weighted_in_adjacecny = np.array([], dtype=float).reshape(0,
                                                                                                                     0); return self._normalized_weighted_in_adjacecny
            D_diag = np.diag(self.in_weighted_degree_matrix)
            D_inv_diag = np.power(D_diag, -1, where=D_diag != 0, out=np.zeros_like(D_diag))
            self._normalized_weighted_in_adjacecny = np.diag(D_inv_diag) @ self.in_weighted_adjacency
        return self._normalized_weighted_in_adjacecny

    def _calculate_digcn_symmetrized_adjacencies(self):
        if self.number_of_nodes == 0:
            empty_tf = tf.constant([], shape=(0, 0), dtype=tf.float32)
            self._degree_normalized_weighted_in_adj_symm = empty_tf
            self._degree_normalized_weighted_out_adj_symm = empty_tf
            return
        A_in_norm = self.degree_normalized_weighted_in_adjacency
        A_out_norm = self.degree_normalized_weighted_out_adjacency
        in_r = (A_in_norm + A_in_norm.T) / 2.
        in_i = (A_in_norm - A_in_norm.T) / 2.
        self._degree_normalized_weighted_in_adj_symm = tf.sqrt(
            tf.square(tf.constant(in_r, tf.float32)) + tf.square(tf.constant(in_i, tf.float32)))
        out_r = (A_out_norm + A_out_norm.T) / 2.
        out_i = (A_out_norm - A_out_norm.T) / 2.
        self._degree_normalized_weighted_out_adj_symm = tf.sqrt(
            tf.square(tf.constant(out_r, tf.float32)) + tf.square(tf.constant(out_i, tf.float32)))

    @property
    def digcn_in_adj_symm(self):
        if self._degree_normalized_weighted_in_adj_symm is None: self._calculate_digcn_symmetrized_adjacencies()
        return self._degree_normalized_weighted_in_adj_symm

    @property
    def digcn_out_adj_symm(self):
        if self._degree_normalized_weighted_out_adj_symm is None: self._calculate_digcn_symmetrized_adjacencies()
        return self._degree_normalized_weighted_out_adj_symm


# --- Operation/Layer Classes ---
class Operation:
    def __init__(self, operation_type: str, name: str, shape: Shape):
        self.name = name;
        self.op_type = operation_type;
        self.shape_spec = shape

    def collect_trainable_parameters(self): return []


class Trainable(Operation):
    def __init__(self, name: str, shape: Shape): super().__init__("Trainable", name, shape)


class NonTrainable(Operation):
    def __init__(self, name: str, shape: Shape): super().__init__("NonTrainable", name, shape)


class BaseActivation(NonTrainable):
    def __init__(self, name: str, input_op: Operation, shape: Shape, act_fn_tf):
        super().__init__(f"Activation:{name}", shape)
        self.input_op = input_op
        self.act_fn_tf = act_fn_tf
        self.output = None

    def compute(self, params=None):
        if self.input_op.output is None: raise ValueError(f"Input for {self.name} is None.")
        self.output = self.act_fn_tf(self.input_op.output)


class Tanh(BaseActivation):
    def __init__(self, input_op: Operation, shape: Shape): super().__init__("Tanh", input_op, shape, tf.nn.tanh)


class Relu(BaseActivation):
    def __init__(self, input_op: Operation, shape: Shape): super().__init__("Relu", input_op, shape, tf.nn.relu)


class Softmax(BaseActivation):
    def __init__(self, input_op: Operation, shape: Shape): super().__init__("Softmax", input_op, shape,
                                                                            lambda x: tf.nn.softmax(x, axis=-1))


class Normalize(NonTrainable):
    def __init__(self, input_op: Operation, shape: Shape):
        super().__init__("Normalize", shape)
        self.input_op = input_op
        self.embedding = None
        self.output = None

    def compute(self, params=None):
        if self.input_op.output is None: raise ValueError("Input for Normalize is None.")
        self.embedding = tf.nn.l2_normalize(self.input_op.output, axis=1)
        self.output = self.input_op.output


class Linear(Trainable):
    def __init__(self, input_op: Operation, shape: Shape, name="Linear"):
        super().__init__(name, shape);
        self.input_op = input_op
        self.W = tf.Variable(Initializers.get("glorot", Shape(shape.in_size, shape.out_size, 0), axis=0),
                             name=f"{name}:W")
        self.b = tf.Variable(Initializers.get("glorot", Shape(1, shape.out_size, 0), axis=1), name=f"{name}:b")
        self.output = None

    def collect_trainable_parameters(self): return [self.W, self.b]

    def compute(self, params=None):
        if self.input_op.output is None: raise ValueError(f"Input for Linear {self.name} is None.")
        W, b = (params[0], params[1]) if params else (self.W, self.b)
        self.output = tf.add(tf.matmul(self.input_op.output, W), b)


class GraphConvolution(Trainable):
    def __init__(self, input_op: Operation, adj: tf.Tensor, shape: Shape, name="GraphConvolution"):
        super().__init__(name, shape)
        self.input_op = input_op
        self.adj = adj
        self.W = tf.Variable(Initializers.get("glorot", Shape(shape.in_size, shape.out_size, 0), axis=0),
                             name=f"{name}:W")
        self.b = tf.Variable(Initializers.get("glorot", Shape(1, shape.out_size, 0), axis=1), name=f"{name}:b")
        self.output = None

    def collect_trainable_parameters(self): return [self.W, self.b]

    def compute(self, params=None):
        if self.input_op.output is None: raise ValueError(f"Input for GCN {self.name} is None.")
        W, b = (params[0], params[1]) if params else (self.W, self.b)
        support = tf.matmul(self.input_op.output, W)
        self.output = tf.sparse.sparse_dense_matmul(self.adj, support) if isinstance(self.adj,
                                                                                     tf.SparseTensor) else tf.matmul(
            self.adj, support)
        self.output = tf.add(self.output, b)


class DirectionalGraphConvolution(Trainable):
    def __init__(self, input_op: Operation, in_adj: tf.Tensor, out_adj: tf.Tensor, shape: Shape, name="DiGCN"):
        super().__init__(name, shape);
        self.input_op = input_op;
        self.in_adj = in_adj;
        self.out_adj = out_adj
        common_w_s = Shape(shape.in_size, shape.out_size, 0);
        common_b_s = Shape(1, shape.out_size, 0);
        scalar_s = Shape(1, 1, 1)
        self.W_all = tf.Variable(Initializers.get("glorot", common_w_s, axis=0), name=f"{name}:W_all")
        self.b_all = tf.Variable(Initializers.get("glorot", common_b_s, axis=1), name=f"{name}:b_all")
        self.W_in = tf.Variable(Initializers.get("glorot", common_w_s, axis=0), name=f"{name}:W_in")
        self.b_in = tf.Variable(Initializers.get("glorot", common_b_s, axis=1), name=f"{name}:b_in")
        self.W_out = tf.Variable(Initializers.get("glorot", common_w_s, axis=0), name=f"{name}:W_out")
        self.b_out = tf.Variable(Initializers.get("glorot", common_b_s, axis=1), name=f"{name}:b_out")
        self.C_in = tf.Variable(Initializers.get("glorot", scalar_s), name=f"{name}:C_in");
        self.C_out = tf.Variable(Initializers.get("glorot", scalar_s), name=f"{name}:C_out")
        self.output = None

    def collect_trainable_parameters(self): return [self.W_all, self.b_all, self.W_in, self.b_in, self.W_out,
                                                    self.b_out, self.C_in, self.C_out]

    def _apply_conv(self, adj, features, W, b, W_err, b_err):
        main = tf.add(tf.matmul(tf.matmul(adj, features), W), b)
        err = tf.add(tf.matmul(tf.matmul(adj, features), W_err), b_err)
        return tf.add(main, err)

    def compute(self, params=None):
        if self.input_op.output is None: raise ValueError(f"Input for DiGCN layer {self.name} is None.")
        p = params if params is not None else self.collect_trainable_parameters()
        W_all, b_all, W_in, b_in, W_out, b_out, C_in, C_out = p
        features = self.input_op.output
        in_c = self._apply_conv(self.in_adj, features, W_in, b_in, W_all, b_all)
        out_c = self._apply_conv(self.out_adj, features, W_out, b_out, W_all, b_all)
        self.output = tf.add(tf.multiply(C_in, in_c), tf.multiply(C_out, out_c))


class Residual(Trainable):
    def __init__(self, input_op1: Operation, input_op2: Operation, shape: Shape, name="Residual"):
        super().__init__(name, shape)
        self.input_op1 = input_op1
        self.input_op2 = input_op2
        self.M = tf.Variable(Initializers.get("glorot", Shape(shape.in_size, shape.out_size, 0), axis=0),
                             name=f"{name}:M")
        self.b = tf.Variable(Initializers.get("glorot", Shape(1, shape.out_size, 0), axis=1), name=f"{name}:b")
        self.output = None

    def collect_trainable_parameters(self): return [self.M, self.b]

    def compute(self, params=None):
        if self.input_op1.output is None or self.input_op2.output is None: raise ValueError(
            f"Input for Residual {self.name} is None.")
        M, b = (params[0], params[1]) if params else (self.M, self.b)
        proj1 = tf.add(tf.matmul(self.input_op1.output, M), b)
        self.output = tf.add(proj1, self.input_op2.output)


# --- Model Base Classes ---
class Model:
    def __init__(self):
        self.hp = {}
        # The rest are typically initialized/managed by BaseGraphModel
        pass

    def prepare(self): pass

    def compile(self): pass

    def compute(self): pass

    def collect(self): pass

    def evaluate(self): pass

    def update(self): pass

    def train(self): pass

    def validate(self): pass

    def visualize(self): pass

    def run(self): pass


class BaseGraphModel(Model):
    def __init__(self, hp: Hyperparameters, graph_data: Graph, model_name: str = "Model"):
        super().__init__()
        self.model_name = model_name
        self.hp = hp
        self.graph_data = graph_data
        self.optimizer = None
        self.train_mask = None
        self.test_mask = None
        self.valid_mask = None
        self.current_predictions = None
        self.current_regularization_loss = tf.constant(0.0, dtype=tf.float32)
        self.current_embedding = None
        self.epoch_losses = []
        self.epoch_accuracies = []
        self.epoch_numbers = []
        self.fold_val_predictions = []
        self.fold_val_losses = []
        self.fold_val_accuracies = []
        self.fold_val_embeddings = []
        self._trainable_layers_list = []
        self._compiled_train_step = None
        self._compiled_test_step = None
        self._compiled_print_info = None

    def _prepare_common(self):
        self.number_of_epochs = self.hp.get('epochs', 100)
        self.split_percent = self.hp.get('split', 70)
        self.learning_rate = self.hp.get('learning_rate', 0.001)
        if self.graph_data.features is None or (
                hasattr(self.graph_data.features, 'shape') and self.graph_data.features.shape[0] == 0):
            default_input_dim = self.hp.get('input_layer_dimension', 64)
            if self.graph_data.number_of_nodes > 0:
                self.graph_data.set_node_features(None, default_dim_if_none=default_input_dim)
            elif self.graph_data.dimensions == 0 and default_input_dim > 0:
                self.graph_data.dimensions = default_input_dim
        self.graph_data.set_node_labels_from_graph_nodes(dummy_community_detection=self.hp.get('dummy_labels', False))
        if self.graph_data.y_labels_one_hot is None or self.graph_data.y_labels_one_hot.shape[0] == 0: print(
            f"Error ({self.model_name}): Labels not populated."); return False
        if self.graph_data.number_of_classes == 0:
            if hasattr(self.graph_data.y_labels_one_hot, 'shape') and self.graph_data.y_labels_one_hot.shape[1] > 0:
                self.graph_data.number_of_classes = self.graph_data.y_labels_one_hot.shape[1]
            else:
                print(f"Error ({self.model_name}): Num classes 0.");
                return False
        self.train_mask, self.test_mask, self.valid_mask = self.graph_data.node_labels_sampler(self.split_percent)
        self.optimizer = Optimizers.optimizer(self.learning_rate)
        return True

    def _build_model_layers(self):
        raise NotImplementedError

    def _collect_all_parameters(self):
        params = [];
        [params.extend(l.collect_trainable_parameters()) for l in self._trainable_layers_list if
         hasattr(l, 'collect_trainable_parameters') and l.collect_trainable_parameters()]
        return params

    def _forward_pass(self, is_training=True):
        raise NotImplementedError

    def _python_train_step_logic(self):
        with tf.GradientTape() as tape:
            all_params = self._collect_all_parameters()
            self._forward_pass(is_training=True)
            current_train_loss, current_train_accuracy = self.evaluate_performance(self.current_predictions,
                                                                                   self.graph_data.y_labels_one_hot,
                                                                                   self.train_mask)
        if not all_params: return current_train_loss, current_train_accuracy
        gradients = tape.gradient(current_train_loss, all_params)
        valid_grads = [(g, p) for i, (g, p) in enumerate(zip(gradients, all_params)) if
                       i < len(gradients) and g is not None]
        if valid_grads: self.optimizer.apply_gradients(valid_grads)
        return current_train_loss, current_train_accuracy

    def _python_test_step_logic(self):
        self._forward_pass(is_training=False)
        return self.evaluate_performance(self.current_predictions, self.graph_data.y_labels_one_hot, self.test_mask)

    def _python_print_info_logic(self, epoch, time_t, lr_t, tr_loss_t, tr_acc_t, te_loss_t, te_acc_t):
        tf.print(self.model_name, "Epoch:", epoch, " Time:",
                 tf.strings.format("{}", [tf.round(time_t * 1000.) / 1000.]), "s", " LR:",
                 tf.strings.format("{}", [tf.round(lr_t * 100000.) / 100000.]), " Train L:",
                 tf.strings.format("{}", [tf.round(tr_loss_t * 10000.) / 10000.]), " Train A:",
                 tf.strings.format("{}", [tf.round(tr_acc_t * 10000.) / 10000.]), " Test L:",
                 tf.strings.format("{}", [tf.round(te_loss_t * 10000.) / 10000.]), " Test A:",
                 tf.strings.format("{}", [tf.round(te_acc_t * 10000.) / 10000.]), summarize=-1)

    def compile_model(self):
        if not self._prepare_common(): raise RuntimeError(f"({self.model_name}) Failed common prep.")
        self._build_model_layers()
        self._compiled_train_step = tf.function(self._python_train_step_logic)
        self._compiled_test_step = tf.function(self._python_test_step_logic)
        self._compiled_print_info = tf.function(self._python_print_info_logic)

    def evaluate_performance(self, predictions, y_true, mask):
        if y_true is None or not hasattr(y_true, 'shape') or y_true.shape[0] == 0 or predictions is None:
            return tf.constant(float('nan')), tf.constant(float('nan'))
        loss = LossFunctions.masked_cross_entropy_loss_evaluater_2(predictions, y_true,
                                                                   mask) + self.current_regularization_loss
        return loss, Evaluation.masked_accuracy_evaluater(predictions, y_true, mask)

    def train_model(self):
        self.epoch_losses = [];
        self.epoch_accuracies = [];
        self.epoch_numbers = []
        lr_t = tf.cast(self.learning_rate, tf.float32)
        for epoch_idx in range(self.number_of_epochs):
            epoch_t = tf.constant(epoch_idx, dtype=tf.int64)
            start_time = time.perf_counter()
            tr_loss, tr_acc = self._compiled_train_step()
            te_loss, te_acc = self._compiled_test_step()
            time_ep = time.perf_counter() - start_time
            time_ep_t = tf.cast(time_ep, tf.float32)
            self._compiled_print_info(epoch_t, time_ep_t, lr_t, tr_loss, tr_acc, te_loss, te_acc)
            self.epoch_losses.append(te_loss.numpy())
            self.epoch_accuracies.append(te_acc.numpy())
            self.epoch_numbers.append(epoch_idx)

    def validate_model(self):
        self._forward_pass(is_training=False)
        if self.graph_data.y_labels_one_hot is None:
            loss_val, acc_val = float('nan'), float('nan')
        else:
            loss_t, acc_t = self.evaluate_performance(self.current_predictions, self.graph_data.y_labels_one_hot,
                                                      self.valid_mask)
            loss_val, acc_val = loss_t.numpy(), acc_t.numpy()
        self.fold_val_predictions.append(
            self.current_predictions.numpy() if self.current_predictions is not None else [])
        self.fold_val_losses.append(loss_val);
        self.fold_val_accuracies.append(acc_val)
        self.fold_val_embeddings.append(self.current_embedding.numpy() if self.current_embedding is not None else [])

    def run_experiment(self):
        all_trial_final_accuracies = []
        self.trial_fold_predictions = []
        self.trial_fold_losses = []
        self.trial_fold_accuracies = []
        self.trial_fold_embeddings = []
        num_trials = self.hp.get("trials", 1)
        k_folds = self.hp.get("K", 1)
        for trial_idx in range(num_trials):
            print(f"\n--- {self.model_name} Trial {trial_idx + 1}/{num_trials} ---")
            current_trial_fold_accuracies = []
            # Reset per-trial accumulators for fold results
            self.trial_fold_predictions = []
            self.trial_fold_losses = []
            self.trial_fold_accuracies = []
            self.trial_fold_embeddings = []

            best_fold_embedding_for_trial = None  # Store embedding of best fold in this trial

            for k_iter in range(k_folds):
                print(f"  K-fold Iteration {k_iter + 1}/{k_folds}")
                try:
                    self.compile_model()
                except RuntimeError as e:
                    print(f"Error compiling {self.model_name} for fold {k_iter + 1}: {e}");
                    continue
                self.train_model();
                self.validate_model()  # validate_model appends to self.fold_val_accuracies

                # Store results from *this specific fold* for trial analysis
                if self.fold_val_accuracies:  # Check if validate_model produced a result for this fold
                    last_fold_acc_this_run = self.fold_val_accuracies[
                        -1]  # Accuracy from the latest validate_model call
                    if not np.isnan(last_fold_acc_this_run):
                        current_trial_fold_accuracies.append(last_fold_acc_this_run)
                        if self.fold_val_embeddings and len(self.fold_val_embeddings) > 0:
                            # Check if current fold's accuracy is the best so far in this trial
                            if best_fold_embedding_for_trial is None or last_fold_acc_this_run > (
                            np.max(current_trial_fold_accuracies[:-1]) if len(
                                    current_trial_fold_accuracies) > 1 else -1):
                                best_fold_embedding_for_trial = self.fold_val_embeddings[-1]

            if current_trial_fold_accuracies:
                best_accuracy_this_trial = np.max(current_trial_fold_accuracies)
                print(
                    f"  Trial {trial_idx + 1} ({self.model_name}) - Best Validation Accuracy in this trial's folds: {best_accuracy_this_trial:.4f}")
                all_trial_final_accuracies.append(float(best_accuracy_this_trial))
                if best_fold_embedding_for_trial is not None and len(best_fold_embedding_for_trial) > 0:
                    self.embedding = best_fold_embedding_for_trial
                    self.visualize_results()  # Visualize using the best embedding of this trial
            else:
                print(f"  Trial {trial_idx + 1} ({self.model_name}) - No successful K-fold results.");
                all_trial_final_accuracies.append(0.0)

        mean_acc = np.mean(all_trial_final_accuracies) if all_trial_final_accuracies else 0.0
        std_acc = np.std(all_trial_final_accuracies) if all_trial_final_accuracies else 0.0
        return {"name": self.model_name, "mean_acc": mean_acc, "std_acc": std_acc,
                "all_trial_acc": all_trial_final_accuracies}

    def visualize_results(self):
        if not self.epoch_numbers or self.embedding is None or (
                isinstance(self.embedding, np.ndarray) and self.embedding.size == 0):
            print(f"({self.model_name}) Not enough data to visualize.")
            return
        try:
            print(f"({self.model_name}) Visualizing model results...")
            embedding_np = np.array(self.embedding);
            if embedding_np.ndim == 1: embedding_np = embedding_np.reshape(-1, 1)
            can_plot_tsne = False
            if embedding_np.shape[0] > 1:
                tsne_perplexity = min(30, max(1, embedding_np.shape[0] - 1))
                if embedding_np.shape[0] > tsne_perplexity and tsne_perplexity > 0:
                    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, learning_rate='auto', init='pca',
                                random_state=123)
                    tsne_transformed = tsne.fit_transform(embedding_np)
                    embedding_df = pd.DataFrame({"x": tsne_transformed[:, 0], "y": tsne_transformed[:, 1]})
                    if self.graph_data.y_labels_one_hot is not None and self.graph_data.y_labels_one_hot.shape[0] == \
                            embedding_np.shape[0]:
                        embedding_df['label'] = tf.argmax(self.graph_data.y_labels_one_hot, axis=1).numpy()
                    can_plot_tsne = True
                else:
                    print(
                        f"({self.model_name}) Samples ({embedding_np.shape[0]}) not sufficient for perplexity ({tsne_perplexity}). Skipping t-SNE.")
            else:
                print(f"({self.model_name}) Not enough samples ({embedding_np.shape[0]}) for t-SNE plot.")
            losses_df = pd.DataFrame({"epochs": self.epoch_numbers, "losses": self.epoch_losses})
            accuracies_df = pd.DataFrame({"epochs": self.epoch_numbers, "accuracies": self.epoch_accuracies})
            num_axes = 2 + (1 if can_plot_tsne else 0)
            fig, axes_arr = plt.subplots(1, num_axes, figsize=(6 * num_axes, 5),
                                         squeeze=False)  # squeeze=False ensures axes_arr is 2D
            axes = axes_arr[0]  # Get the 1D array of axes

            fig.suptitle(f'{self.model_name} Training Visualization')
            plot_idx = 0
            if can_plot_tsne:
                sns.scatterplot(ax=axes[plot_idx], data=embedding_df, x='x', y='y',
                                hue='label' if 'label' in embedding_df else None, palette='viridis',
                                legend='full' if 'label' in embedding_df else False);
                axes[plot_idx].set_title(f'Embeddings (t-SNE)')
                plot_idx += 1
            sns.lineplot(ax=axes[plot_idx], data=losses_df, x='epochs', y='losses')
            axes[plot_idx].set_title(f'Test Loss/Epoch')
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Loss')
            plot_idx += 1
            sns.lineplot(ax=axes[plot_idx], data=accuracies_df, x='epochs', y='accuracies')
            axes[plot_idx].set_title(f'Test Acc/Epoch')
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Accuracy')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        except Exception as e:
            print(f"({self.model_name}) Error during visualization: {e}");

            traceback.print_exc()


class GCN(BaseGraphModel):
    def __init__(self, hp, graph_data: UndirectedGraph):
        super().__init__(hp, graph_data, model_name="GCN")

    def _build_model_layers(self):
        self._trainable_layers_list = [];
        self.current_regularization_loss = tf.constant(0.0, dtype=tf.float32)
        input_d = self.graph_data.dimensions
        if input_d == 0: input_d = self.hp.get('input_layer_dimension', 32); print(
            f"Warning ({self.model_name}): Graph dimensions 0. Using HP dim: {input_d}")
        l1_d = self.hp.get('layer_1_dimension', 16);
        l2_d = self.hp.get('layer_2_dimension', 16)
        out_d = self.graph_data.number_of_classes
        if out_d == 0: out_d = self.hp.get('output_layer_dimension', self.hp.get('layer_2_dimension', 2)); print(
            f"Warning ({self.model_name}): Classes 0. Output dim to {out_d}.")
        reg_r = self.hp.get('regularization_rate', 0.0005)
        if self.graph_data.degree_normalized_adjacency is None or self.graph_data.degree_normalized_adjacency.size == 0: raise ValueError(
            f"({self.model_name}) NormAdj invalid.")
        self.adj_matrix_tf = tf.constant(self.graph_data.degree_normalized_adjacency, dtype=tf.float32)
        if self.graph_data.features is None or self.graph_data.features.shape[0] == 0: raise ValueError(
            f"({self.model_name}) Features invalid.")
        self.input_features_op = InputOp(input_tensor=self.graph_data.features, normalize=False)
        self.gcn1 = GraphConvolution(input_op=self.input_features_op, adj=self.adj_matrix_tf,
                                     shape=Shape(input_d, l1_d, 0), name="GCN1");
        self._trainable_layers_list.append(self.gcn1)
        self.act1 = Tanh(input_op=self.gcn1, shape=Shape(l1_d, l1_d, 0))
        self.gcn2 = GraphConvolution(input_op=self.act1, adj=self.adj_matrix_tf, shape=Shape(l1_d, l2_d, 0),
                                     name="GCN2");
        self._trainable_layers_list.append(self.gcn2)
        self.act2 = Tanh(input_op=self.gcn2, shape=Shape(l2_d, l2_d, 0))
        self.output_linear = Linear(input_op=self.act2, shape=Shape(l2_d, out_d, 0), name="OutputLinear");
        self._trainable_layers_list.append(self.output_linear)
        if reg_r > 0:
            for layer in self._trainable_layers_list:
                for param in layer.collect_trainable_parameters(): self.current_regularization_loss += tf.nn.l2_loss(
                    param) * reg_r

    def _forward_pass(self, is_training=True):
        self.input_features_op.output = self.graph_data.features
        self.gcn1.compute()
        self.act1.compute()
        self.gcn2.compute()
        self.act2.compute()
        self.output_linear.compute()
        self.current_predictions = self.output_linear.output
        self.current_embedding = self.act2.output


class CustomDiGCN(BaseGraphModel):
    def __init__(self, hp, graph_data: DirectedGraph):
        super().__init__(hp, graph_data, model_name="CustomDiGCN")

    def _build_model_layers(self):
        self._trainable_layers_list = [];
        self.current_regularization_loss = tf.constant(0.0, dtype=tf.float32)
        input_d = self.graph_data.dimensions
        if input_d == 0: input_d = self.hp.get('input_layer_dimension', 32); print(
            f"Warning ({self.model_name}): Graph dimensions 0. Using HP dim: {input_d}")
        l1_d = self.hp.get('layer_1_dimension', 16);
        l2_d = self.hp.get('layer_2_dimension', 16)
        out_d = self.graph_data.number_of_classes
        if out_d == 0: out_d = self.hp.get('output_layer_dimension', self.hp.get('layer_2_dimension', 2)); print(
            f"Warning ({self.model_name}): Classes 0. Output dim to {out_d}.")
        reg_r = self.hp.get('regularization_rate', 0.0005)
        if self.graph_data.digcn_in_adj_symm is None or self.graph_data.digcn_out_adj_symm is None: raise ValueError(
            f"({self.model_name}) DiGCN Adj invalid.")
        self.in_adj_tf = self.graph_data.digcn_in_adj_symm;
        self.out_adj_tf = self.graph_data.digcn_out_adj_symm
        if self.graph_data.features is None or self.graph_data.features.shape[0] == 0: raise ValueError(
            f"({self.model_name}) Features invalid.")
        self.input_features_op = InputOp(input_tensor=self.graph_data.features, normalize=True)
        self.digcn1 = DirectionalGraphConvolution(input_op=self.input_features_op, in_adj=self.in_adj_tf,
                                                  out_adj=self.out_adj_tf, shape=Shape(input_d, l1_d, 0),
                                                  name="DiGCN1");
        self._trainable_layers_list.append(self.digcn1)
        self.act1 = Tanh(input_op=self.digcn1, shape=Shape(l1_d, l1_d, 0))
        self.digcn2 = DirectionalGraphConvolution(input_op=self.act1, in_adj=self.in_adj_tf, out_adj=self.out_adj_tf,
                                                  shape=Shape(l1_d, l2_d, 0), name="DiGCN2");
        self._trainable_layers_list.append(self.digcn2)
        self.act2 = Tanh(input_op=self.digcn2, shape=Shape(l2_d, l2_d, 0))
        self.output_linear = Linear(input_op=self.act2, shape=Shape(l2_d, out_d, 0), name="OutputLinear");
        self._trainable_layers_list.append(self.output_linear)
        if reg_r > 0:
            for layer in self._trainable_layers_list:
                for param in layer.collect_trainable_parameters(): self.current_regularization_loss += tf.nn.l2_loss(
                    param) * reg_r

    def _forward_pass(self, is_training=True):
        self.input_features_op.output = self.graph_data.features
        self.digcn1.compute()
        self.act1.compute()
        self.digcn2.compute()
        self.act2.compute()
        self.output_linear.compute()
        self.current_predictions = self.output_linear.output
        self.current_embedding = self.act2.output


# --- Data Loading ---
def load_planetoid_dataset_as_custom_graph(dataset_name: str, data_root_path: str = 'data/planetoid'):
    if not PYG_AVAILABLE:
        print(f"Cannot load {dataset_name}: PyTorch Geometric is not available.")
        return None, None
    print(f"Loading Planetoid dataset: {dataset_name} from {data_root_path}")
    try:
        planetoid_dataset = Planetoid(root=data_root_path, name=dataset_name.capitalize())
        pyg_data = planetoid_dataset[0]
    except Exception as e:
        print(f"Error loading Planetoid dataset {dataset_name}: {e}");
        return None, None

    num_nodes = pyg_data.num_nodes
    node_features_np = pyg_data.x.numpy()
    node_labels_np = pyg_data.y.numpy()
    edge_list_directed = []
    if pyg_data.edge_index is not None:
        for i in range(pyg_data.edge_index.shape[1]):
            u, v = pyg_data.edge_index[0, i].item(), pyg_data.edge_index[1, i].item()
            edge_list_directed.append((str(u), str(v), 1.0, 0))  # Using str for node IDs
    nodes_dict_orig = {str(i): node_labels_np[i].item() for i in range(num_nodes)}

    print(f"  Creating UndirectedGraph for {dataset_name}...")
    undirected_graph = UndirectedGraph(nodes=deepcopy(nodes_dict_orig), edges=deepcopy(edge_list_directed))
    undirected_graph.set_node_features(node_features_np.copy())

    print(f"  Creating DirectedGraph for {dataset_name}...")
    directed_graph = DirectedGraph(nodes=deepcopy(nodes_dict_orig), edges=deepcopy(edge_list_directed))
    directed_graph.set_node_features(node_features_np.copy())

    print(
        f"  {dataset_name} loaded: {undirected_graph.number_of_nodes} nodes, {len(edge_list_directed)} raw directed edges.")
    return undirected_graph, directed_graph


def load_data_karate_undirected():
    G_nx = nx.karate_club_graph();
    raw_edges = [(str(e[0]), str(e[1]), 1.0, 1.0) for e in G_nx.edges()]
    raw_nodes = {str(n): G_nx.nodes[n]['club'] for n in G_nx.nodes()}
    return UndirectedGraph(nodes=raw_nodes, edges=raw_edges)


def load_data_karate_directed():
    G_nx = nx.karate_club_graph();
    raw_edges = [(str(e[0]), str(e[1]), 1.0, 1.0) for e in G_nx.edges()]
    raw_nodes = {str(n): G_nx.nodes[n]['club'] for n in G_nx.nodes()}
    return DirectedGraph(nodes=raw_nodes, edges=raw_edges)


# --- Comparison Table ---
def display_comparison_table(model_results_list):
    if not model_results_list: print("No model results to display."); return
    print("\n\n--- Model Comparison Summary ---")
    df = pd.DataFrame(model_results_list)
    df = df[['name', 'mean_acc', 'std_acc']]  # Ensure column order
    df.rename(columns={'name': 'Model Name', 'mean_acc': 'Mean Accuracy', 'std_acc': 'Std Dev Acc'}, inplace=True)

    # Format float columns
    df['Mean Accuracy'] = df['Mean Accuracy'].map('{:.4f}'.format)
    df['Std Dev Acc'] = df['Std Dev Acc'].map('{:.4f}'.format)

    print(df.to_string(index=False))
    # Example of how to add more metrics if run_experiment returned them:
    # df['Mean F1'] = df['Mean F1'].map('{:.4f}'.format)
    # print(df.to_markdown(index=False)) # For markdown output


# --- Main Execution ---
def main():
    print("Refactored Model Testing")
    all_model_results = []
    # For creating data directory

    dataset_name = "PubMed"  # Options: "Cora", "CiteSeer", "PubMed", "karate"
    planetoid_data_root = "data/planetoid"
    Path(planetoid_data_root).mkdir(parents=True, exist_ok=True)

    ud_graph, d_graph = None, None
    common_hp_overrides = {}
    default_dim = 64

    if dataset_name.lower() == "karate":
        ud_graph = load_data_karate_undirected()
        d_graph = load_data_karate_directed()

        common_hp_overrides = {"dummy_labels": True, "input_layer_dimension": default_dim,
                               "layer_1_dimension": default_dim, "layer_2_dimension": default_dim}
    elif dataset_name.lower() in ["cora", "citeseer", "pubmed"]:
        if not PYG_AVAILABLE: print(f"Cannot run on {dataset_name}: PyTorch Geometric missing."); return
        ud_graph, d_graph = load_planetoid_dataset_as_custom_graph(dataset_name=dataset_name,
                                                                   data_root_path=planetoid_data_root)
        common_hp_overrides = {"dummy_labels": False, "input_layer_dimension": default_dim,
                               "layer_1_dimension": default_dim, "layer_2_dimension": default_dim}
    else:
        print(f"Dataset {dataset_name} not configured.")
        return

    if ud_graph is None or d_graph is None: print(f"Failed to load {dataset_name}. Exiting."); return

    base_hp_values = {
        "trials": 1, "K": 10, "epochs": 500,
        "split": 60, "regularization_rate": 0.0005, "learning_rate": 0.01,
        "dropout_rate": 0.5,
    }
    base_hp_values.update(common_hp_overrides)

    # GCN
    hpGCN = Hyperparameters()
    hpGCN.update(base_hp_values)
    if ud_graph.dimensions > 0:
        hpGCN.add('input_layer_dimension', ud_graph.dimensions)
    else:
        ud_graph.set_node_features(None,
                                   default_dim_if_none=hpGCN.get('input_layer_dimension'))  # Ensure features are set

    print(f"\n--- Testing GCN Model on {dataset_name} ---")
    model_gcn = GCN(hpGCN, ud_graph)
    results_gcn = model_gcn.run_experiment()
    all_model_results.append(results_gcn)

    # Custom DiGCN
    hpCustomDiGCN = Hyperparameters()
    hpCustomDiGCN.update(base_hp_values)
    if d_graph.dimensions > 0:
        hpCustomDiGCN.add('input_layer_dimension', d_graph.dimensions)
    else:
        d_graph.set_node_features(None, default_dim_if_none=hpCustomDiGCN.get('input_layer_dimension'))

    print(f"\n--- Testing Custom DiGCN Model on {dataset_name} ---")
    model_custom_digcn = CustomDiGCN(hpCustomDiGCN, d_graph)
    results_custom_digcn = model_custom_digcn.run_experiment()
    all_model_results.append(results_custom_digcn)

    display_comparison_table(all_model_results)


if __name__ == '__main__':
    main()
