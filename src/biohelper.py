import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import h5py
import re
import random
import multiprocessing
import math
import chardet
import pickle
import gc
import tracemalloc
import tempfile
import logging
import requests
import csv
from typing import Union, Tuple, Optional, List, Dict, Iterator, Any, Callable

from operator import itemgetter
from copy import deepcopy
from contextlib import redirect_stdout
from typing import Optional, Any

import collections.abc
from collections import defaultdict
from collections import Counter
from collections import deque

from itertools import chain
from tqdm import tqdm

import pandas as pd
import pyarrow as pa
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import numpy as np
import scipy as sp
from scipy.stats import wilcoxon, pearsonr
import networkx as nx
import mysql.connector

from gensim.models import Word2Vec

import sklearn as skl
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ndcg_score
)

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.python.keras.backend import dtype

from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
from Levenshtein import distance, ratio

import matplotlib.pyplot as plt
import seaborn as sns

from Bio import SeqIO

random.seed(123)
np.random.seed(123)
np.set_printoptions(threshold=np.inf)
tf.random.set_seed(123)
tf.config.run_functions_eagerly(False)
tf.config.threading.set_inter_op_parallelism_threads(32)

print("Numpy Version: ", np.version.version)
print("Tensorflow Version: ", tf.__version__)
print("Keras Version: ", tf.keras.__version__)

print(tf.config.list_physical_devices('GPU'))


class Graph:
    def __init__(self, nodes={"A": [1.0]}, edges=[("A", "A", 1.0, 1.0)]):
        self.nodes = nodes

        self.original_edges = edges
        self.edges = edges

        self.node_indices()
        self.edge_indices()

        # Check to see if all the nodes and all the edges match
        self.integrity_check()

    def integrity_check(self):
        print("Graph integrity check")
        weightless_edges = []
        for e in self.edges:
            weightless_edges.append((e[0], e[1]))
        data = nx.Graph(weightless_edges)
        connected_components = nx.connected_components(data)
        if len(list(connected_components)) > 1:
            print("WARNING: Graph has more than one component")
        else:
            print("Graph has no components")
        all_nodes = []
        for e in self.edges:
            s_n = e[0]
            t_n = e[1]
            all_nodes.append(s_n)
            all_nodes.append(t_n)
        all_nodes_from_edges = sorted(list(set(all_nodes)))
        input_nodes = list(self.nodes.keys())
        input_nodes.sort()
        all_nodes_from_nodes = input_nodes
        if all_nodes_from_nodes != all_nodes_from_edges:
            print("WARNING: nodes and edges are not equal")
        else:
            print("Nodes matches edges")
        number_of_edges = len(self.edges)
        complete_edges = (self.number_of_nodes * (self.number_of_nodes - 1)) / 2
        check = False
        if number_of_edges == complete_edges:
            s = 0
            for e in self.edges:
                s = s + e[2]
            if s == number_of_edges:
                check = True
        if check:
            print("Graph is complete and full")
        else:
            print("WARNING: Graph is incomplete")

    def adjacency_probability_test(self, adjacency):
        print(np.sum(adjacency, axis=1))

    def node_indices(self):
        self.node_index = {}
        self.node_inverted_index = {}
        if len(self.nodes) == 0:
            all_nodes = []
            for pair in self.edges:
                source_node = str(pair[0])
                target_node = str(pair[1])
                all_nodes.append(source_node)
                all_nodes.append(target_node)
            all_nodes = list(set(all_nodes))
            for i in range(len(all_nodes)):
                self.nodes[i] = "1"
                self.node_index[all_nodes[i]] = i
                self.node_inverted_index[i] = all_nodes[i]
        else:
            for i, n in enumerate(self.nodes.keys()):
                self.node_index[n] = i
                self.node_inverted_index[i] = n
            new_nodes = {}
            for n in self.nodes:
                new_nodes[self.node_index[n]] = self.nodes[n]
            self.nodes = new_nodes
        self.number_of_nodes = len(self.nodes.keys())

    def edge_indices(self):
        indexed_edge_list = []
        for es in self.edges:
            indexed_edge_list.append((self.node_index[es[0]], self.node_index[es[1]], es[2], es[3]))
        self.edges = indexed_edge_list
        self.number_of_edges = len(self.edges)
        del indexed_edge_list

    def node_features(self, features, dim):
        if features is not None:
            # coo = features.tocoo()
            # Convert to COO format
            row, col, data = [], [], []
            for i, row_data in enumerate(features):
                for j, value in enumerate(row_data):
                    if value != 0:
                        row.append(i)
                        col.append(j)
                        data.append(value)

            # Create a COO matrix
            coo = sp.sparse.coo_matrix((data, (row, col)))
            indices = np.mat([coo.row, coo.col]).transpose()
            tf_sparse_tensor = tf.SparseTensor(indices, coo.data, coo.shape)
            self.features = tf.cast(tf.sparse.to_dense(tf_sparse_tensor), tf.float32)
            self.dimensions = self.features.shape[1]
        else:
            class Feature:
                def __init__(self, n, d):
                    self.output = Initializers.identity(Shape(in_size=d, out_size=d, batch_size=n))

            self.features = Feature(self.number_of_nodes, dim).output
            self.dimensions = self.features.shape[1]

    def node_labels(self, dummy):
        if not dummy:
            self.node_label_profile = {}
            self.number_of_classes = 0
            self.node_label_profile = defaultdict(list)
            for key, val in sorted(self.nodes.items()):
                if val != "N/A":
                    self.node_label_profile[val].append(key)
            self.node_label_profile = dict(self.node_label_profile)
            self.number_of_classes = int(len(self.node_label_profile.keys()))
        else:
            self.node_labels = {}
            self.node_label_profile = {}
            self.number_of_classes = 0
            weightless_edges = []
            for e in self.edges:
                weightless_edges.append((e[0], e[1]))
            data = nx.Graph(weightless_edges)
            louvain_communities = nx.community.louvain_communities(data)

            if len(list(louvain_communities)) < 2:
                print("WARNING: Graph has only one Louvain community")
            c = 0
            for community in louvain_communities:
                c = c + 1
                for n in community:
                    self.node_labels[n] = str(c)
                    self.nodes[n] = str(c)
            del data
            del weightless_edges[:]
            del louvain_communities[:]
            gc.collect()
            self.node_label_profile = defaultdict(list)
            for key, val in sorted(self.node_labels.items()):
                self.node_label_profile[val].append(key)
            self.node_label_profile = dict(self.node_label_profile)
            self.number_of_classes = int(len(self.node_label_profile.keys()))

    def node_labels_sampler(self, split_percent):
        nodes = list(self.nodes.keys())
        k = np.floor((split_percent * len(nodes)) / 100)
        number_of_classes = len(list(self.node_label_profile.keys()))
        sample_label_size = np.floor(k / number_of_classes)
        final_samples = []
        for labels, ns in self.node_label_profile.items():
            number_of_samples = 0
            if sample_label_size > len(ns):
                number_of_samples = len(ns)
            elif sample_label_size <= len(ns):
                number_of_samples = sample_label_size
            samples = list(np.random.choice(ns, size=int(number_of_samples), replace=False))
            final_samples = final_samples + samples
        train_samples = list(set(final_samples))
        rest_samples = list(set(nodes) - set(train_samples))
        middle_index = np.math.floor(len(rest_samples) / 2)
        valid_samples = rest_samples[0:middle_index]
        test_samples = rest_samples[middle_index:]
        train_samples_indices = []
        for t_s in train_samples:
            train_samples_indices.append(t_s)
        valid_samples_indices = []
        for t_s in valid_samples:
            valid_samples_indices.append(t_s)
        test_samples_indices = []
        for t_s in test_samples:
            test_samples_indices.append(t_s)
        train_mask = np.zeros((len(nodes)), dtype=int)
        train_mask[train_samples_indices] = 1
        train_mask = train_mask.astype(bool)
        valid_mask = np.zeros((len(nodes)), dtype=int)
        valid_mask[valid_samples_indices] = 1
        valid_mask = valid_mask.astype(bool)
        test_mask = np.zeros((len(nodes)), dtype=int)
        test_mask[test_samples_indices] = 1
        test_mask = test_mask.astype(bool)
        return train_mask, test_mask, valid_mask

    def second_degree_graph(self):
        new_edges = []
        new_nodes = {}
        new_node_id = 0
        sorted(self.edges, key=itemgetter(0))
        for e in self.edges:
            s_n = str(e[0])
            t_n = str(e[1])
            new_edge = s_n + "|" + t_n
            l = int(e[3])
            new_node_id = new_node_id + 1
            if s_n < t_n:
                new_nodes[new_edge] = l
        for n1, info1 in new_nodes.items():
            nodes1 = set(n1.split("|"))
            for n2, info2 in new_nodes.items():
                if n1 == n2:
                    continue
                nodes2 = set(n2.split("|"))
                intersect = list(nodes1.intersection(nodes2))
                if len(intersect) > 0:
                    new_edges.append((n1, n2, float(1), float(1)))
        return new_edges, new_nodes

    def to_networkx(self):
        # Generated by Gemini
        G = nx.Graph()
        G.add_weighted_edges_from(self.edges)
        return G

    def wl(self, graph2):
        # Generated by Gemini
        graph1 = self.to_networkx()
        graph2 = graph2.to_networkx()
        # Initialize the node labels.
        node_labels1 = {node: node for node in graph1.nodes()}
        node_labels2 = {node: node for node in graph2.nodes()}
        # Iterate until the node labels converge.
        while True:
            # Update the node labels.
            for node in graph1.nodes():
                node_labels1[node] = (node_labels1[node],
                                      tuple(node_labels1[neighbor] for neighbor in graph1.neighbors(node)))
            for node in graph2.nodes():
                node_labels2[node] = (node_labels2[node],
                                      tuple(node_labels2[neighbor] for neighbor in graph2.neighbors(node)))
            # Check if the node labels have converged.
            if node_labels1 == node_labels2:
                return True  # If the node labels have not converged, continue iterating.
        return False

    def create_edge_embeddings(self,
                               interaction_pairs: list[tuple[str, str, int]],
                               embeddings: dict[str, np.ndarray],
                               method: str = 'concatenate'
                               ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        print(f"Creating edge embeddings using method: {method}...")
        edge_features_list = []
        labels_list = []
        skipped_pairs_missing_embeddings = 0
        skipped_pairs_mismatched_dim = 0

        if not embeddings:
            print("Warning: Embeddings dictionary is empty. Cannot create edge features.")
            return None, None

        embedding_dim = 0
        for pid in embeddings:  # Find first valid embedding to determine dimension
            if isinstance(embeddings[pid], np.ndarray) and embeddings[
                pid].ndim > 0:  # ensure it's an array and has a shape
                if embeddings[pid].shape[0] > 0:  # ensure dimension is not zero
                    embedding_dim = embeddings[pid].shape[0]
                    break

        if embedding_dim == 0:
            print(
                "Warning: Could not determine a valid embedding dimension from provided protein embeddings (all embeddings might be empty or not found).")
            return None, None
        print(f"Inferred embedding dimension: {embedding_dim}")

        for p1_id, p2_id, label in interaction_pairs:
            emb1 = embeddings.get(p1_id)
            emb2 = embeddings.get(p2_id)

            if emb1 is not None and emb2 is not None:
                if emb1.shape[0] != embedding_dim or emb2.shape[0] != embedding_dim:
                    skipped_pairs_mismatched_dim += 1
                    continue

                if method == 'concatenate':
                    edge_emb = np.concatenate((emb1, emb2))
                elif method == 'average':
                    edge_emb = (emb1 + emb2) / 2.0
                elif method == 'hadamard':
                    edge_emb = emb1 * emb2
                elif method == 'subtract':
                    edge_emb = np.abs(emb1 - emb2)
                else:
                    raise ValueError(f"Unknown edge embedding method: {method}")
                edge_features_list.append(edge_emb)
                labels_list.append(label)
            else:
                skipped_pairs_missing_embeddings += 1

        print(f"Created {len(edge_features_list)} edge features.")
        if skipped_pairs_missing_embeddings > 0:
            print(f"Skipped {skipped_pairs_missing_embeddings} pairs due to one or both protein embeddings not found.")
        if skipped_pairs_mismatched_dim > 0:
            print(f"Skipped {skipped_pairs_mismatched_dim} pairs due to mismatched embedding dimensions.")

        if not edge_features_list:
            print("No edge features were created. Check protein IDs, embedding files, and dimension consistency.")
            return None, None

        return np.array(edge_features_list, dtype=np.float32), np.array(labels_list, dtype=np.int32)


class BipartiteGraph(Graph):
    def __init__(self, edges=[], nodes={}):
        super().__init__(edges=edges, nodes=nodes)
        self.inverted_node_memberships = Algorithms.invert_dict(self.nodes)
        self.type1_nodes = list(list(self.inverted_node_memberships.values())[0])
        self.type2_nodes = list(list(self.inverted_node_memberships.values())[1])
        if len(list(self.inverted_node_memberships.values())) > 2:
            print("WARNING: Graph has more than 2 node types")
        self.type1_memberships = []
        self.type2_memberships = []
        for n in self.type1_nodes:
            self.type1_memberships.append(self.node_index[n])
        for n in self.type2_nodes:
            self.type2_memberships.append(self.node_index[n])
        # Dimensions of the matrix
        self.type2_count = len(self.type2_nodes)
        self.type1_count = len(self.type1_nodes)
        self.biadjacency_matrix()
        self.degree_matrix()
        self.degree_normalized_biadjacency_matrix()

    def biadjacency_matrix(self):
        # Index dictionaries for the matrix. Note that this set of indices is different of that in the condor object (that one is for the igraph network.)
        self.type1_index = {self.type1_nodes[i]: i for i in range(0, self.type1_count)}
        self.type2_index = {self.type2_nodes[i]: i for i in range(0, self.type2_count)}
        # Computes weighted biadjacency matrix.
        self.biadjacency = np.matrix(np.zeros((self.type2_count, self.type1_count)))
        for edge in self.edges:
            self.biadjacency[int(self.type2_index[edge[0]]), int(self.type1_index[edge[1]])] = float(edge[2])

    def degree_matrix(self):
        self.type2_degree = np.array(np.sum(self.biadjacency, axis=0))
        self.type1_degree = np.array(np.sum(self.biadjacency, axis=1))
        self.combined_degrees = np.matrix(self.type1_degree @ self.type2_degree)

    def degree_normalized_biadjacency_matrix(self):
        self.degree_normalized_biadjacency = np.divide(self.biadjacency, self.combined_degrees)

    def bipartite_modularity_clustering(self):
        # Modified BRIMS algorithm
        # Computes sum of edges and bimodularity matrix.
        m = float(sum(self.type1_degree))
        B = self.biadjacency - (self.combined_degrees / m)
        # Computation of initial modularity matrix for tar and reg nodes from the membership dataframe.
        T_ed = list(zip([self.type2_index[j] for j in [i for i in self.type2_nodes]], self.type2_memberships))
        T0 = np.zeros((self.type2_count, self.n))
        for edge in T_ed:
            T0[edge] = 1
        R_ed = list(zip([self.type1_index[j] for j in [i for i in self.type1_nodes]], self.type1_memberships))
        R0 = np.zeros((self.type1_count, self.n))
        for edge in R_ed:
            R0[edge] = 1
        deltaQmin = min(1 / m, 1e-5)
        Qnow = 0
        deltaQ = 1
        while (deltaQ > deltaQmin):
            # Right sweep
            Tp = T0.transpose().dot(B)
            R = np.zeros((self.type1_count, self.n))
            am = np.array(np.argmax(Tp.transpose(), axis=1))
            for i in range(0, len(am)):
                R[i, am[i][0]] = 1
            # Left sweep
            Rp = B.dot(R)
            T = np.zeros((self.type2_count, self.n))
            am = np.array(np.argmax(Rp, axis=1))
            for i in range(0, len(am)):
                T[i, am[i][0]] = 1
            T0 = T
            Qthen = Qnow
            RtBT = T.transpose().dot(B.dot(R))
            Qcoms = (1 / m) * (np.diagonal(RtBT))
            Qnow = sum(Qcoms)
            deltaQ = Qnow - Qthen

        # Character Communities
        tar_memberships = list(zip(list(self.type2_index), [T[i, :].argmax() for i in range(0, len(self.type2_index))]))
        type2_memberships = {}
        for tup in tar_memberships:
            type2_memberships[str(tup[0])] = int(tup[1])
        grouped_type2_memberships = defaultdict(list)
        for index, row in sorted(type2_memberships.items()):
            grouped_type2_memberships[row].append(index)
        type2_components = []
        for k, v in grouped_type2_memberships.items():
            type2_components.append(v)

        # Entity Communities
        entity_memberships = list(
            zip(list(self.type1_index), [R[i, :].argmax() for i in range(0, len(self.type1_index))]))
        type1_memberships = {}
        for tup in entity_memberships:
            type1_memberships[str(tup[0])] = int(tup[1])
        grouped_type1_memberships = defaultdict(list)
        for index, row in sorted(type1_memberships.items()):
            grouped_type1_memberships[row].append(index)
        type1_components = []
        for k, v in grouped_type1_memberships.items():
            type1_components.append(v)

        return type1_components, type2_components


class UndirectedGraph(Graph):
    def __init__(self, edges=[], nodes={}):
        super().__init__(edges=edges, nodes=nodes)
        self.adjacency = self.adjacency_matrix()
        self.weighted_adjacency = self.weighted_adjacency_matrix()
        self.degree_matrix = self.degree_matrix()
        self.weighted_degree_matrix = self.weighted_degree_matrix()
        self.degree_normalized_adjacency = self.degree_normalized_adjacency_matrix()
        self.degree_weighted_normalized_adjacency = self.degree_weighted_normalized_adjacency_matrix()

    def __str__(self):
        return "An undirected graph of " + str(self.number_of_nodes) + " nodes and " + str(
            self.number_of_edges) + " edges."

    def adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
        for oes in self.edges:
            s_n = int(oes[0])
            t_n = int(oes[1])
            adjacency_matrix[s_n, t_n] = 1
            adjacency_matrix[t_n, s_n] = 1
        return adjacency_matrix

    def weighted_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
        for oes in self.edges:
            s_n = int(oes[0])
            t_n = int(oes[1])
            w = float(oes[2])
            adjacency_matrix[s_n, t_n] = w
            adjacency_matrix[t_n, s_n] = w
        return adjacency_matrix

    def degree_matrix(self):
        return np.diag(np.count_nonzero(self.adjacency, axis=0))

    def weighted_degree_matrix(self):
        return np.diag(np.sum(self.weighted_adjacency, axis=0))

    def degree_normalized_adjacency_matrix(self):
        self_connections = self.adjacency + np.identity(self.number_of_nodes)
        inverse_degree = np.linalg.inv(self.degree_matrix)
        return np.array(np.multiply(self_connections, inverse_degree))

    def degree_weighted_normalized_adjacency_matrix(self):
        self_connections = self.weighted_adjacency + np.identity(self.number_of_nodes)
        inverse_degree = np.linalg.inv(self.weighted_degree_matrix)
        return np.array(np.multiply(self_connections, inverse_degree))


class DirectedGraph(Graph):
    def __init__(self, edges=[], nodes={}):
        super().__init__(edges=edges, nodes=nodes)
        self.in_indexed_edges = Algorithms.flip_list(self.edges)
        self.out_adjacency_matrix, self.in_adjacency_matrix = self.adjacency_matrices()
        self.out_weighted_adjacency_matrix, self.in_weighted_adjacency_matrix = self.weighted_adjacency_matrices()
        self.out_degree_matrix, self.in_degree_matrix = self.degree_matrices()
        self.out_weighted_degree_matrix, self.in_weighted_degree_matrix = self.weighted_degree_matrices()
        self.normalized_out_adjacecny, self.normalized_in_adjacecny = self.degree_normalized_directed_adjacency_matrices()
        self.normalized_weighted_out_adjacecny, self.normalized_weighted_in_adjacecny = self.degree_normalized_weighted_directed_adjacency_matrices()
        self.degree_normalized_adjacency = self.degree_normalized_adjacency()
        # self.degree_normalized_weighted_in_adjacency, self.degree_normalized_weighted_out_adjacency = self.convert_to_symmetric_degree_normalized_directed_adjacency_matrices()
        self.degree_normalized_weighted_in_adjacency, self.degree_normalized_weighted_out_adjacency = self.symmetric_degree_normalized_directed_adjacencies()

    def symmetric_degree_normalized_directed_adjacencies(self):
        in_real_part = (np.add(self.normalized_weighted_in_adjacecny,
                               np.transpose(self.normalized_weighted_in_adjacecny)) / 2)
        in_imaginary_part = (np.subtract(self.normalized_weighted_in_adjacecny,
                                         np.transpose(self.normalized_weighted_in_adjacecny)) / 2)
        # in_adj_decompostion = in_real_part + in_imaginary_part
        # test_in_equivalence = Utils.are_matrices_identical(self.normalized_weighted_in_adjacecny,
        #                                                    in_adj_decompostion)

        in_adjacency_approximation = tf.sqrt(tf.add(tf.square(tf.convert_to_tensor(in_real_part, dtype=tf.float32)),
                                                    tf.square(
                                                        tf.convert_to_tensor(in_imaginary_part, dtype=tf.float32))))

        out_real_part = (np.add(self.normalized_weighted_out_adjacecny,
                                np.transpose(self.normalized_weighted_out_adjacecny)) / 2)
        out_imaginary_part = (np.subtract(self.normalized_weighted_out_adjacecny,
                                          np.transpose(self.normalized_weighted_out_adjacecny)) / 2)

        out_adjacency_approximation = tf.sqrt(tf.add(tf.square(tf.convert_to_tensor(out_real_part, dtype=tf.float32)),
                                                     tf.square(
                                                         tf.convert_to_tensor(out_imaginary_part, dtype=tf.float32))))
        # out_adj_decompostion = out_real_part + out_imaginary_part
        # test_out_equivalence = Utils.are_matrices_identical(self.normalized_weighted_out_adjacecny,
        #                                                     out_adj_decompostion)

        # if test_in_equivalence and test_out_equivalence:
        #     return in_real_part, out_real_part
        # else:
        #     return -1
        return in_adjacency_approximation, out_adjacency_approximation

    def symmetric_degree_normalized_directed_adjacency_matrices(self):
        in_real_part = (np.add(self.normalized_weighted_in_adjacecny,
                               np.transpose(self.normalized_weighted_in_adjacecny)) / 2)
        in_imaginary_part = (np.subtract(self.normalized_weighted_in_adjacecny,
                                         np.transpose(self.normalized_weighted_in_adjacecny)) / 2)
        in_adj_decompostion = in_real_part + in_imaginary_part
        # test_in_equivalence = Utils.are_matrices_identical(self.normalized_weighted_in_adjacecny,
        #                                                    in_adj_decompostion)

        out_real_part = (np.add(self.normalized_weighted_out_adjacecny,
                                np.transpose(self.normalized_weighted_out_adjacecny)) / 2)
        out_imaginary_part = (np.subtract(self.normalized_weighted_out_adjacecny,
                                          np.transpose(self.normalized_weighted_out_adjacecny)) / 2)
        out_adj_decompostion = out_real_part + out_imaginary_part
        # test_out_equivalence = Utils.are_matrices_identical(self.normalized_weighted_out_adjacecny,
        #                                                     out_adj_decompostion)

        # if test_in_equivalence and test_out_equivalence:
        #     return in_real_part, out_real_part
        # else:
        #     return -1
        return in_real_part, out_real_part

    def degree_normalized_adjacency(self):
        in_real_part = (np.add(self.normalized_weighted_in_adjacecny,
                               np.transpose(self.normalized_weighted_in_adjacecny)) / 2)
        in_imaginary_part = (np.subtract(self.normalized_weighted_in_adjacecny,
                                         np.transpose(self.normalized_weighted_in_adjacecny)) / 2)
        in_adj_decompostion = in_real_part + in_imaginary_part
        test_in_equivalence = Math.are_matrices_identical(self.normalized_weighted_in_adjacecny, in_adj_decompostion)

        out_real_part = (np.add(self.normalized_weighted_out_adjacecny,
                                np.transpose(self.normalized_weighted_out_adjacecny)) / 2)
        out_imaginary_part = (np.subtract(self.normalized_weighted_out_adjacecny,
                                          np.transpose(self.normalized_weighted_out_adjacecny)) / 2)
        out_adj_decompostion = out_real_part + out_imaginary_part
        test_out_equivalence = Math.are_matrices_identical(self.normalized_weighted_out_adjacecny, out_adj_decompostion)

        if test_in_equivalence and test_out_equivalence:
            degree_normalized_adjacency = np.add(in_real_part, out_real_part)
            return degree_normalized_adjacency
        else:
            return -1

    def __str__(self):
        return "An directed graph of " + str(self.number_of_nodes) + " nodes and " + str(
            self.number_of_edges) + " edges."

    def to_networkx(self):
        # Generated by Gemini
        G = nx.DiGraph()
        G.add_weighted_edges_from(self.edges)
        return G

    def adjacency_matrices(self):
        in_adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
        out_adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
        for oes in self.edges:
            out_adjacency_matrix[oes[0], oes[1]] = 1
        for ies in self.in_indexed_edges:
            in_adjacency_matrix[ies[0], ies[1]] = 1
        return out_adjacency_matrix, in_adjacency_matrix

    def weighted_adjacency_matrices(self):
        in_weighted_adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
        out_weighted_adjacency_matrix = np.zeros((self.number_of_nodes, self.number_of_nodes), dtype=float)
        for oes in self.edges:
            out_weighted_adjacency_matrix[oes[0], oes[1]] = oes[2]
        for ies in self.in_indexed_edges:
            in_weighted_adjacency_matrix[ies[0], ies[1]] = ies[2]
        return out_weighted_adjacency_matrix, in_weighted_adjacency_matrix

    def degree_matrices(self):
        in_real_part = (np.add(self.in_adjacency_matrix, np.transpose(self.in_adjacency_matrix)) / 2)
        in_imaginary_part = (np.subtract(self.in_adjacency_matrix, np.transpose(self.in_adjacency_matrix)) / 2)
        in_adj_decompostion = in_real_part + in_imaginary_part
        test_in_equivalence = Math.are_matrices_identical(self.in_adjacency_matrix, in_adj_decompostion)

        out_real_part = (np.add(self.out_adjacency_matrix, np.transpose(self.out_adjacency_matrix)) / 2)
        out_imaginary_part = (np.subtract(self.out_adjacency_matrix, np.transpose(self.out_adjacency_matrix)) / 2)
        out_adj_decompostion = out_real_part + out_imaginary_part
        test_out_equivalence = Math.are_matrices_identical(self.out_adjacency_matrix, out_adj_decompostion)

        if test_in_equivalence and test_out_equivalence:
            return np.diag(np.count_nonzero(out_real_part, axis=0)), np.diag(np.count_nonzero(in_real_part, axis=0))
        else:
            return np.zeros(np.shape(out_real_part)), np.zeros((in_real_part))

    def weighted_degree_matrices(self):
        in_real_part = (np.add(self.in_weighted_adjacency_matrix, np.transpose(self.in_weighted_adjacency_matrix)) / 2)
        in_imaginary_part = (
                np.subtract(self.in_weighted_adjacency_matrix, np.transpose(self.in_weighted_adjacency_matrix)) / 2)
        in_adj_decompostion = in_real_part + in_imaginary_part
        test_in_equivalence = Math.are_matrices_identical(self.in_weighted_adjacency_matrix, in_adj_decompostion)

        out_real_part = (
                np.add(self.out_weighted_adjacency_matrix, np.transpose(self.out_weighted_adjacency_matrix)) / 2)
        out_imaginary_part = (np.subtract(self.out_weighted_adjacency_matrix,
                                          np.transpose(self.out_weighted_adjacency_matrix)) / 2)
        out_adj_decompostion = out_real_part + out_imaginary_part
        test_out_equivalence = Math.are_matrices_identical(self.out_weighted_adjacency_matrix, out_adj_decompostion)

        if test_in_equivalence and test_out_equivalence:
            return np.diag(np.sum(out_real_part, axis=0)), np.diag(np.sum(in_real_part, axis=0))
        else:
            return np.zeros(np.shape(out_real_part)), np.zeros((in_real_part))

    def degree_normalized_directed_adjacency_matrices(self):
        return np.matmul(self.out_adjacency_matrix, np.linalg.inv(self.out_degree_matrix)), np.matmul(
            self.in_adjacency_matrix, np.linalg.inv(self.in_degree_matrix))

    def degree_normalized_weighted_directed_adjacency_matrices(self):
        return np.matmul(self.out_weighted_adjacency_matrix, np.linalg.inv(self.out_weighted_degree_matrix)), np.matmul(
            self.in_weighted_adjacency_matrix, np.linalg.inv(self.in_weighted_degree_matrix))


# --- Configuration ---
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "adminadmin",
    "database": "pdi"
}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


class Shape:
    def __init__(self, in_size, out_size, batch_size):
        self.in_size = in_size
        self.out_size = out_size
        self.batch_size = batch_size

    def __str__(self):
        return "(In: " + str(self.in_size) + ", Out: " + str(self.out_size) + ", Batch: " + str(self.batch_size) + ")"


class Input:
    def __init__(self, input, normalize):
        self.input = input
        if normalize:
            norms = tf.norm(self.input, axis=1, keepdims=True)
            self.output = self.input / norms
        else:
            self.output = self.input


class Hyperparameters(dict):
    def __init__(self):
        pass

    def add(self, key: str, value):
        self[key] = value

    def get(self, key: str, default=None):
        if key not in self:
            ratios = {}
            for k in self.keys():
                r = ratio(key, k)
                ratios[r] = k
            max_ratio = max(ratios.keys())
            if max_ratio > 0.7:
                return ratios[max_ratio]
        return self[key]


class Utils:
    def __init__(self):
        pass


class Math(Utils):
    def __init__(self):
        super().__init__()

    @staticmethod
    def cosine_distance(vector1, vector2):
        return sp.spatial.distance.cosine(vector1, vector2)

    @staticmethod
    def are_matrices_identical(A, B):
        """ Generated by Gemini
        Compares two matrices and returns True if they are identical, False otherwise.

        Args:
          A: A list of lists representing the first matrix.
          B: A list of lists representing the second matrix.

        Returns:
          True if the two matrices are identical, False otherwise.
        """

        # Check if the dimensions of the two matrices are equal.
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            return False

        # Compare each element of both matrices.
        for i in range(len(A)):
            for j in range(len(A[0])):
                if A[i][j] != B[i][j]:
                    return False

        # If all elements are equal, return True.
        return True

    @staticmethod
    def raise_matrix_to_power(m, a):
        k = a
        A = m
        result = tf.Variable(A)

        i = tf.constant(0)
        c = lambda i: tf.less(i, k - 1)

        def body(i):
            result.assign(tf.matmul(A, result))
            return [tf.add(i, 1)]

        _ = tf.while_loop(c, body, [i])

        return result

    @staticmethod
    def jaccard(list1, list2):
        # Gemini generated
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(set(list1)) + len(set(list2))) - intersection
        result = float(intersection) / union
        if result > 0.5:
            return True
        else:
            return False


class Statistics(Utils):
    def __init__(self):
        super().__init__()

    @staticmethod
    def t_test(model1, model2):
        statistic, p_value = sp.stats.shapiro(model1)
        print("shapiro-statistic for Model 1 = " + str(statistic))
        print("p-value = " + str(p_value))
        statistic, p_value = sp.stats.shapiro(model2)
        print("shapiro-statistic for Model 2 = " + str(statistic))
        print("p-value = " + str(p_value))
        t_stat, p_value = sp.stats.ttest_rel(model1, model2, alternative="less")
        print("t-statistic = " + str(t_stat))
        print("p-value = " + str(p_value))


class MachineLearning(Utils):
    def __init__(self):
        super().__init__()


class LossFunctions(MachineLearning):
    def __init__(self):
        super().__init__()

    @staticmethod
    def masked_cross_entropy_loss_evaluater_1(predictions, y, mask):
        term1 = tf.boolean_mask(predictions, mask)
        term2 = tf.math.log(term1)
        term3 = tf.boolean_mask(y, mask)
        term4 = term3 * term2
        term5 = tf.reduce_sum(term4)
        term6 = -1 * tf.reduce_mean(term5)
        return term6

    @staticmethod
    def masked_cross_entropy_loss_evaluater_2(predictions, y, mask):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)


class Optimizers(MachineLearning):
    def __init__(self):
        super().__init__()

    # Optimizers
    @staticmethod
    def optimizer(learn_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
        return optimizer


class Evaluation(MachineLearning):
    def __init__(self):
        super().__init__()

    # Evaluation from Kipf et al 2017
    @staticmethod
    def masked_accuracy_evaluater(prediction, y, mask):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        accuracy = tf.reduce_mean(accuracy_all)
        return accuracy


class Initializers(MachineLearning):
    def __init__(self):
        super().__init__()

    @staticmethod
    def identity(shape: Shape, axis=1):
        init = tf.keras.initializers.Identity()
        if axis == 1:
            return init(shape=(shape.batch_size, shape.out_size), dtype=tf.float32)
        return init(shape=(shape.in_size, shape.out_size), dtype=tf.float32)

    @staticmethod
    def uniform(shape: Shape, axis=1):
        init = tf.keras.initializers.RandomUniform(0, 1)
        if axis == 1:
            return init(shape=(shape.batch_size, shape.out_size), dtype=tf.float32)
        return init(shape=(shape.in_size, shape.out_size), dtype=tf.float32)

    @staticmethod
    def glorot(shape: Shape, axis=1):
        init = tf.keras.initializers.GlorotUniform()
        if axis == 1:
            return init(shape=(shape.batch_size, shape.out_size), dtype=tf.float32)
        return init(shape=(shape.in_size, shape.out_size), dtype=tf.float32)

    @staticmethod
    def zeros(shape: Shape, axis=1):
        init = tf.keras.initializers.zeros()
        if axis == 1:
            return init(shape=(shape.batch_size, shape.out_size), dtype=tf.float32)
        return init(shape=(shape.in_size, shape.out_size), dtype=tf.float32)

    @staticmethod
    def ones(shape: Shape, axis=1):
        init = tf.keras.initializers.ones()
        if axis == 1:
            return init(shape=(shape.batch_size, shape.out_size), dtype=tf.float32)
        return init(shape=(shape.in_size, shape.out_size), dtype=tf.float32)


class Algorithms(Utils):
    def __init__(self):
        super().__init__()

    # Algorithms
    @staticmethod
    def invert_dict(d):
        inverted_dict = {}
        for key, value_list in d.items():
            for value in value_list:
                inverted_dict.setdefault(value, set()).add(key)
        return inverted_dict

    @staticmethod
    def reverse_dict(d):
        reversed_dict = defaultdict(list)
        for key, value in d.items():
            reversed_dict[value].append(key)
        return reversed_dict

    @staticmethod
    def flip_list(list):
        flipped_edges = []
        for es in list:
            flipped_edges.append((es[1], es[0], es[2]))
        return flipped_edges

    @staticmethod
    def flatten_extend(matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list

    @staticmethod
    def print_list(l, filename):
        with open(filename, 'w') as fp:
            for item in l:
                # write each item on a new line
                fp.write("%s\n" % item)
            print('Done')

    @staticmethod
    def object_equality(o1, o2):
        if o1 == o2:
            return True
        else:
            return False

    @staticmethod
    def sample_mask(idx, l):
        """Create mask."""
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)

    @staticmethod
    def encode_characters(character_sequence):
        ascii_sequence = []
        for c in character_sequence:
            ascii_sequence.append(str(ord(c)))
        return ascii_sequence

    @staticmethod
    def strip_non_alphanumeric(text):
        """Removes all non-alphanumeric characters from a string.

        Args:
          text: The string to strip.

        Returns:
          The stripped string.
        """
        return re.sub(r'[^a-zA-Z0-9]', '', text.rstrip().lstrip())

    @staticmethod
    def compute_char_edges_for_sequence(sequence_text: str,
                                        strip_algo: Callable[[str], str] = strip_non_alphanumeric,
                                        encode_algo: Callable[[List[str]], List[int]] = encode_characters
                                        ) -> List[Dict[str, Union[str, float]]]:
        if not sequence_text or not isinstance(sequence_text, str): return []
        main_body = strip_algo(sequence_text)
        if not main_body: return []
        final_character_walk: List[str] = list(main_body)
        final_character_walk.append(" ")
        encoded_walk: List[int] = encode_algo(final_character_walk)
        if len(encoded_walk) < 2: return []
        edges = [f"{encoded_walk[i]}|{encoded_walk[i + 1]}" for i in range(len(encoded_walk) - 1)]
        if not edges: return []
        edge_counter = Counter(edges)
        character_edges: List[Dict[str, Union[str, float]]] = []
        for edge_str, count in edge_counter.items():
            parts = edge_str.split("|")
            if len(parts) == 2:
                try:
                    character_edges.append({'from_node': parts[0], 'to_node': parts[1], 'weight': float(count)})
                except ValueError:
                    logger.warning(f"Could not parse edge components {parts[0]}, {parts[1]} for edge '{edge_str}'")
            else:
                logger.warning(f"Malformed edge string '{edge_str}' found.")
        return character_edges


class FileOps(Utils):
    def __init__(self):
        super().__init__()

    @staticmethod
    def determine_file_encoding(input_file_name):
        encoding = None
        with open(input_file_name, "rb") as rawdata:
            result = chardet.detect(rawdata.read())
            encoding = result["encoding"]
            if encoding == "ascii":
                encoding = "utf-8"
        return encoding

    @staticmethod
    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    @staticmethod
    def parse_generic_tsv(filepath: str, delimiter: str = '\t', skip_header_lines: int = 0) -> Iterator[List[str]]:
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as fh:
                reader = csv.reader(fh, delimiter=delimiter)
                for _ in range(skip_header_lines):
                    try:
                        next(reader)
                    except StopIteration:
                        logger.warning(f"File '{filepath}' has < {skip_header_lines} lines.")
                        return
                for row in reader: yield row
        except FileNotFoundError:
            logger.error(f"File not found: '{filepath}'")
        except Exception as e:
            logger.error(f"Error parsing TSV file '{filepath}': {e}")

    @staticmethod
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(filename):
        with open(filename + '.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_checkpoint(obj_name, filename):
        print(f"Attempting to save the list to '{filename}'...")
        try:
            FileOps.save_obj(obj_name, filename)
            print(f"Successfully saved the list to '{filename}'")
        except Exception as e:
            print(f"An error occurred while saving: {e}")

        # --- Part 2: Deserializing and Loading the List ---
        print(f"\nAttempting to load the list from '{filename}'...")
        if os.path.exists(filename):
            try:
                loaded_obj = FileOps.load_obj(filename)
                print(f"Successfully loaded the list from '{filename}'")
            except Exception as e:
                print(f"An error occurred while loading: {e}")
        else:
            print(f"Error: The file '{filename}' was not found.")

    @staticmethod
    def load_h5_embeddings(file_path: str) -> dict[str, np.ndarray]:
        """
        Loads protein embeddings from an HDF5 file.
        Assumes HDF5 structure: keys are protein IDs, values are embedding vectors.
        """
        print(f"Loading embeddings using 'load_h5_embeddings' from: {file_path}...")
        embeddings: dict[str, np.ndarray] = {}
        if not os.path.exists(file_path):
            print(f"Error: Embedding file not found at {file_path}")
            return embeddings
        try:
            with h5py.File(file_path, 'r') as hf:
                for protein_id in hf.keys():
                    embeddings[protein_id] = np.array(hf[protein_id][:], dtype=np.float32)
            print(f"Loaded {len(embeddings)} embeddings from {file_path}.")
        except Exception as e:
            print(f"Error loading HDF5 embeddings from {file_path}: {e}")
        return embeddings

    @staticmethod
    def load_custom_embeddings(file_path: str) -> dict[str, np.ndarray]:
        """
        Loads custom protein embeddings. Adapts based on file extension or internal logic.
        Modify this to suit the format of your embeddings file.
        Expected output: a dictionary {protein_id: embedding_vector (np.ndarray)}
        """
        print(f"Attempting to load embeddings using 'load_custom_embeddings' from: {file_path}...")
        embeddings: dict[str, np.ndarray] = {}
        if not os.path.exists(file_path):
            print(f"Error: Custom embedding file not found at {file_path}")
            return embeddings

        if file_path.endswith('.npz'):
            print(f"Detected .npz extension, attempting NPZ loading logic for {file_path}...")
            try:
                data = np.load(file_path, allow_pickle=True)
                if 'embeddings_dict' in data:
                    loaded_item = data['embeddings_dict'].item()
                    if isinstance(loaded_item, dict):
                        embeddings = {str(k): np.array(v, dtype=np.float32) for k, v in loaded_item.items()}
                    else:
                        print(f"NPZ 'embeddings_dict' in {file_path} did not contain a dictionary.")
                elif 'ids' in data and 'vectors' in data:
                    ids = data['ids']
                    vectors = data['vectors']
                    embeddings = {str(pid): vec.astype(np.float32) for pid, vec in zip(ids, vectors)}
                else:
                    # Fallback for simple dict stored as the first item in npz
                    if data.files:
                        loaded_obj = data[data.files[0]]
                        if isinstance(loaded_obj.item(), dict):
                            embeddings = {str(k): np.array(v, dtype=np.float32) for k, v in loaded_obj.item().items()}
                        else:
                            print(
                                f"Could not interpret NPZ file structure in {file_path}. Expected 'embeddings_dict' or 'ids'/'vectors' keys, or a single dict item.")
                    else:
                        print(f"NPZ file {file_path} is empty or has no recognizable data structure.")
                print(f"Loaded {len(embeddings)} custom embeddings from NPZ {file_path}.")
            except Exception as e:
                print(f"Error loading custom NPZ embeddings from {file_path}: {e}")
            return embeddings
        elif file_path.endswith('.h5'):  # Delegate to HDF5 loader if it's an H5 file
            print(f"Detected .h5 extension for {file_path}, delegating to 'load_h5_embeddings'.")
            return FileOps.load_h5_embeddings(file_path)
        else:
            print(f"Warning: 'load_custom_embeddings' does not have specific logic for file type of {file_path}. "
                  "Please adapt this function or ensure the file is an .npz or .h5 handled by default.")
            # TODO: Add your custom loading logic here for other formats if needed
            return {}


class DatabaseOps(Utils):  # Unchanged
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        try:
            conn_test = mysql.connector.connect(host=self.db_config["host"], user=self.db_config["user"],
                                                password=self.db_config["password"])
            cursor_test = conn_test.cursor()
            cursor_test.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_config['database']}")
            logger.info(f"Database '{self.db_config['database']}' ensured to exist.")
            cursor_test.close();
            conn_test.close()
        except mysql.connector.Error as err:
            logger.error(f"Failed to connect/ensure database '{self.db_config['database']}': {err}")

    def _get_connection(self) -> mysql.connector.MySQLConnection:
        try:
            conn = mysql.connector.connect(**self.db_config);
            return conn  # type: ignore
        except mysql.connector.Error as err:
            logger.error(f"Database connection error: {err}");
            raise

    def execute_ddl(self, ddl_statement: str) -> bool:
        conn = None;
        cursor = None
        try:
            conn = self._get_connection();
            cursor = conn.cursor()
            for stmt in ddl_statement.split(';'):
                if stmt.strip(): cursor.execute(stmt.strip())
            conn.commit();
            logger.info(f"Successfully executed DDL: {ddl_statement.splitlines()[0][:100]}...")
            return True
        except mysql.connector.Error as err:
            logger.error(f"Error executing DDL '{ddl_statement.splitlines()[0][:100]}...': {err}")
            if conn: conn.rollback(); return False
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()

    def insert_data_batch(self, table_name: str, columns: List[str],
                          values_batch: List[Tuple[Any, ...]]) -> bool:
        if not values_batch: return True
        conn = None;
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join(f"`{col}`" for col in columns)
            sql = f"INSERT INTO `{table_name}` ({column_names}) VALUES ({placeholders})"
            cursor.executemany(sql, values_batch)
            conn.commit()
            logger.info(f"{cursor.rowcount} record(s) inserted into {table_name}.")
            return True
        except mysql.connector.Error as err:
            logger.error(f"DB batch insert error into {table_name}: {err}")
            if conn: conn.rollback(); return False
        except Exception as e:
            logger.error(f"Unexpected error during batch insert into {table_name}: {e}")
            if conn: conn.rollback(); return False
        finally:
            if cursor: cursor.close()
            if conn and conn.is_connected(): conn.close()


class ProteinAlgorithms(Algorithms):  # Unchanged
    @staticmethod
    def load_sequences_with_dask(filepath: str, sequence_column_name: str = "Sequence",
                                 blocksize: str = '128MB', dtype_for_sequence_col: str = 'str'
                                 ) -> Optional[dd.DataFrame]:
        if not os.path.exists(filepath): logger.error(f"File not found at {filepath}"); return None
        logger.info(f"Initiating Dask to process file: {filepath} with blocksize: {blocksize}")
        try:
            ddf = dd.read_csv(filepath, sep='\t', usecols=[sequence_column_name],
                              blocksize=blocksize, dtype={sequence_column_name: dtype_for_sequence_col})
            logger.info(
                f"Successfully created Dask DataFrame for column '{sequence_column_name}'. Partitions: {ddf.npartitions}")
            return ddf
        except ValueError as ve:
            if f"'{sequence_column_name}'" in str(ve) and "is not in columns" in str(ve):
                logger.error(f"Column '{sequence_column_name}' not found in {filepath}.")
            else:
                logger.error(f"ValueError during Dask loading: {ve}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during Dask loading: {e}");
            return None

    # @staticmethod
    # def compute_sequence_lengths_dask(sequences_series: dd.Series) -> Optional[dd.Series]:
    #     if not isinstance(sequences_series, dd.Series): logger.error("Input must be a Dask Series."); return None
    #     return sequences_series.str.len()

    @staticmethod
    def sequences_to_edges_partitioned(sequences: List[str]
                                       ) -> List[List[Dict[str, Union[str, float]]]]:
        all_sequence_edges: List[List[Dict[str, Union[str, float]]]] = []
        for seq_text in sequences:
            all_sequence_edges.append(Algorithms.compute_char_edges_for_sequence(seq_text))
        return all_sequence_edges

    @staticmethod
    def sequences_to_ids_partitioned(sequences: List[str], graph: Graph,
                                     strip_algo: Callable[[str], str] = Algorithms.strip_non_alphanumeric,
                                     encode_algo: Callable[[List[str]], List[int]] = Algorithms.encode_characters,
                                     ) -> List[List[Dict[str, Union[str, float]]]]:
        def replace_with_ids(x):
            return graph.node_index[x]

        all_sequence_ids: List[List[Dict[str, Union[str, float]]]] = []
        for sequence_text in sequences:
            if not sequence_text or not isinstance(sequence_text, str): return []
            main_body = strip_algo(sequence_text)
            if not main_body: return []
            final_character_walk: List[str] = list(main_body)
            final_character_walk.append(" ")
            encoded_walk: List[int] = encode_algo(final_character_walk)
            if len(encoded_walk) < 2: return []
            all_sequence_ids = [list(map(replace_with_ids, i)) for i in range(len(encoded_walk) - 1)]
        return all_sequence_ids

    @staticmethod
    def get_protein_id(protein_name: str) -> Optional[str]:
        if not protein_name or not isinstance(protein_name, str): return None
        return protein_name.split('_')[0]

    @staticmethod
    def get_protein_name_from_id(protein_id: str) -> Optional[str]:
        if not protein_id or not isinstance(protein_id, str): return None
        return f"{protein_id}_protein"

    @staticmethod
    def download_pdb_structure(pdb_id: str, output_dir: str = ".", file_format: str = "pdb",
                               timeout: int = 30
                               ) -> Tuple[Optional[str], Optional[str]]:
        if file_format not in ['pdb', 'cif']: logger.error(f"Invalid file format '{file_format}'."); return None, None
        download_url = f"https://files.rcsb.org/download/{pdb_id.lower()}.{file_format}"
        logger.info(f"Attempting to download: {download_url}")
        try:
            response = requests.get(download_url, timeout=timeout)
            response.raise_for_status()
            pdb_data = response.text
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"{pdb_id.lower()}.{file_format}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pdb_data)
            logger.info(f"Structure saved to: {output_path}")
            return output_path, pdb_data
        except requests.exceptions.HTTPError as he:
            if he.response.status_code == 404:
                logger.error(f"PDB ID '{pdb_id}' format '{file_format}' not found (404).")
            else:
                logger.error(f"HTTP error for {pdb_id}: {he.response.status_code} - {he.response.text[:200]}")
            return None, None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {pdb_id}: {e}");
            return None, None

    @staticmethod
    def dask_partition_processor(partition_df: pd.DataFrame, graph: Graph) -> pd.DataFrame:
        if 'Sequence' not in partition_df.columns:
            logger.warning("Partition missing 'Sequence' column.")
            for col, dtype in [('Sequence_Length', 'int64'), ('Sequence_Edges', 'object')]:
                if col not in partition_df: partition_df[col] = pd.Series(dtype=dtype)
            return partition_df
        if partition_df.empty:
            for col, dtype in [('Sequence_Length', 'int64'), ('Sequence_Edges', 'object')]:
                if col not in partition_df: partition_df[col] = pd.Series(dtype=dtype)
            return partition_df
        sequences_series = partition_df['Sequence'].fillna("").astype(str)
        partition_df['Sequence_Length'] = sequences_series.str.len().astype('int64')
        partition_df['Sequence_Edges'] = ProteinAlgorithms.sequences_to_edges_partitioned(sequences_series.tolist())
        partition_df['Sequence_List'] = ProteinAlgorithms.sequences_to_ids_partitioned(sequences_series.tolist(), graph)
        return partition_df


class ProteinFileOps(FileOps):  # Unchanged
    @staticmethod
    def parse_fasta_efficiently(file_path: str) -> Iterator[Dict[str, Any]]:
        try:
            with open(file_path, 'r') as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    yield {'identifier': record.id, 'description': record.description,
                           'length': len(record.seq), 'sequence': str(record.seq)}
        except FileNotFoundError:
            logger.error(f"FASTA file not found: '{file_path}'")
        except Exception as e:
            logger.error(f"Error parsing FASTA file '{file_path}': {e}")

    @staticmethod
    def parse_fasta_to_dict(file_path: str, default_source: str = "UNKNOWN") -> Dict[str, Dict[str, Any]]:
        sequences: Dict[str, Dict[str, Any]] = {}
        current_id: Optional[str] = None
        current_seq_lines: List[str] = []
        current_description: str = ""
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line: continue
                    if line.startswith('>'):
                        if current_id:
                            sequences[current_id] = {'source': default_source, 'description': current_description,
                                                     'length': len("".join(current_seq_lines)),
                                                     'sequence': "".join(current_seq_lines)}
                        header_parts = line[1:].split(maxsplit=1)
                        current_id = header_parts[0]
                        current_description = header_parts[1] if len(header_parts) > 1 else ""
                        current_seq_lines = []
                    elif current_id:
                        current_seq_lines.append(line)
                if current_id:
                    sequences[current_id] = {'source': default_source, 'description': current_description,
                                             'length': len("".join(current_seq_lines)),
                                             'sequence': "".join(current_seq_lines)}
        except FileNotFoundError:
            logger.error(f"FASTA file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error parsing FASTA file to dict '{file_path}': {e}")
        return sequences

    @staticmethod
    def parse_uniref_fasta_to_tsv_rows(input_fasta_path: str) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(input_fasta_path): logger.error(f"Input FASTA missing: '{input_fasta_path}'"); return
        logger.info(f"Starting parsing of UniRef FASTA: '{input_fasta_path}'")
        try:
            for record in ProteinFileOps.parse_fasta_efficiently(input_fasta_path):
                desc = record['description']
                org, tax, rep = "N/A", "N/A", "N/A"
                if m := re.search(r"OS=(.*?)(?:\s(?:OX=|GN=|PE=|SV=)|$)", desc): org = m.group(1).strip()
                if m := re.search(r"OX=(\d+)", desc): tax = m.group(1).strip()
                if m := re.search(r"RepID=([\S]+)", desc): rep = m.group(1).strip()
                yield {'UniRef_ID': record['identifier'], 'Organism': org,
                       'Sequence_Length': record['length'], 'Tax_ID': tax,
                       'Rep_ID': rep, 'Sequence': record['sequence']}
        except Exception as e:
            logger.error(f"An error occurred during UniRef FASTA parsing: {e}")

    @staticmethod
    def parse_fasta(filepath):
        """
        Parses a FASTA file and extracts sequence data and descriptions.

        Args:
            filepath: The path to the FASTA file.

        Returns:
            A list of tuples, where each tuple contains (description, sequence).
            Returns an empty list if there's an issue with the file.
        """
        try:
            records = []
            for record in SeqIO.parse(filepath, "fasta"):
                description = record.description  # Header information
                sequence = str(record.seq)  # The protein sequence
                records.append((description, sequence))
            return records
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return []
        except Exception as e:  # Catching potential other Bio.SeqIO parse errors.
            print(f"An error occurred while parsing the file: {e}")
            return []

    @staticmethod
    def load_interaction_pairs(file_path: str, label: int) -> list[tuple[str, str, int]]:
        print(f"Loading interaction pairs from: {file_path} (label: {label})...")
        pairs: list[tuple[str, str, int]] = []
        skipped_lines = 0
        if not os.path.exists(file_path):
            print(f"Error: Interaction file not found at {file_path}")
            return pairs
        try:
            # Explicitly set encoding, good practice.
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    current_line_stripped = line.strip()

                    # Header check logic:
                    if i == 0:
                        # Ensure we are operating on a string for these checks
                        if isinstance(current_line_stripped, str) and (
                                'protein' in current_line_stripped.lower() or 'interactor' in current_line_stripped.lower() or current_line_stripped.startswith(
                            '#')):
                            print(f"Skipping header line: {current_line_stripped}")
                            continue

                    # Skip empty lines after header check
                    if not current_line_stripped:
                        # It's an empty line, not necessarily malformed if it's not the header
                        # but we can choose to count it or not. Let's not count it as skipped for now
                        # unless it causes issues in splitting.
                        continue

                    # Proceed with processing the line, using current_line_stripped
                    parts = current_line_stripped.split('\t')
                    if len(parts) < 2:
                        parts = current_line_stripped.split(',')  # Try comma if tab split fails

                    if len(parts) >= 2:
                        # Strip individual protein IDs in case of leading/trailing whitespace
                        p1 = parts[0].strip()
                        p2 = parts[1].strip()
                        if p1 and p2:  # Ensure both protein IDs are non-empty after stripping
                            pairs.append((p1, p2, label))
                        else:
                            # One of the protein IDs became empty after stripping
                            if current_line_stripped:  # Only count if original line wasn't just whitespace
                                skipped_lines += 1
                    elif current_line_stripped:  # Line was not empty but couldn't be split into at least 2 parts
                        skipped_lines += 1

        except Exception as e:
            print(f"CRITICAL Error during processing of {file_path}: {e}")
            import traceback
            print("Detailed traceback of the error:")
            traceback.print_exc()  # This will print the exact line where the error occurred
            # The function will return the pairs loaded so far, or an empty list if error was early.

        if skipped_lines > 0:
            print(f"Note: Skipped {skipped_lines} malformed or empty-ID lines in {file_path}.")
        # Moved this message to be outside the try-except, so it always reports what was loaded.
        print(f"Successfully loaded {len(pairs)} pairs from {file_path}.")
        return pairs

    @staticmethod
    def save_dask_dataframe_to_parquet(dask_df: dd.DataFrame, directory_path: str,
                                       engine: str = 'pyarrow', overwrite: bool = True,
                                       compression: Optional[str] = 'gzip',
                                       schema: Optional[pa.Schema] = None) -> bool:  # Added schema parameter
        if dask_df is None: logger.error("Dask DataFrame is None. Nothing to save."); return False
        if not isinstance(dask_df, dd.DataFrame): logger.error("Input is not a Dask DataFrame."); return False
        if dask_df.npartitions == 0:
            logger.warning("Attempting to save an empty Dask DataFrame (0 partitions). Skipping save to Parquet.")
            return False

        logger.info(
            f"Saving Dask DataFrame to Parquet at: {directory_path} using compression: {compression}, engine: {engine}")
        try:
            # Dask's to_parquet with overwrite=True should handle directory.
            # No need for manual os.makedirs here if relying on Dask.
            kwargs_for_to_parquet = {
                "write_index": False,
                "engine": engine,
                "overwrite": overwrite,
                "compression": compression
            }
            if engine == 'pyarrow' and schema is not None:
                kwargs_for_to_parquet['schema'] = schema
                logger.info("Using explicit PyArrow schema for saving.")
            elif schema is not None and engine != 'pyarrow':
                logger.warning(
                    f"Schema provided but engine is '{engine}', not 'pyarrow'. Schema might not be used by engine.")

            dask_df.to_parquet(directory_path, **kwargs_for_to_parquet)
            logger.info(f"Successfully saved Dask DataFrame to {directory_path}")
            return True
        except Exception as e:
            # The error message already contains "Failed to convert partition to expected pyarrow schema"
            logger.error(f"Error saving Dask DataFrame to Parquet: {e}", exc_info=True)
            return False

    @staticmethod
    def load_dask_dataframe_from_parquet(directory_path: str, columns: Optional[List[str]] = None,
                                         engine: str = 'pyarrow'
                                         ) -> Optional[dd.DataFrame]:
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            logger.error(f"Parquet directory not found or not a dir: {directory_path}")
            return None
        logger.info(f"Loading Dask DataFrame from Parquet at: {directory_path}")
        try:
            ddf = dd.read_parquet(directory_path, columns=columns, engine=engine)
            logger.info(f"Successfully loaded Dask DataFrame from {directory_path}")
            if ddf is not None: logger.info(f"  Columns: {ddf.columns.tolist()}, Partitions: {ddf.npartitions}")
            return ddf
        except Exception as e:
            if "smaller than the minimum file footer" in str(e) or "Invalid Parquet file size" in str(e):
                logger.error(f"Error loading Parquet: File in {directory_path} likely empty/corrupt. {e}")
            else:
                logger.error(f"Error loading Dask DataFrame from Parquet: {e}", exc_info=True)
            return None


def tests():
    logger.info("Starting example workflow...")
    run_suffix = f"_{int(time.time())}"

    # base_data_dir = os.path.join("G:", "My Drive", "Knowledge", "Research", "TWU", "Projects", "Link Prediction in Protein Interaction Networks via Structural Sequence Embedding", "Data")
    base_data_dir = "G:/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/"
    # base_data_dir = os.path.join(".", "sample_data_output") # For local testing

    for example_subdir_part in ["protein.sequences.v12.0.fa", "uniprot_sprot.fasta", "ProteinNegativeAssociations",
                                "uniref100", "dask_checkpoints"]:
        os.makedirs(os.path.join(base_data_dir, example_subdir_part), exist_ok=True)
    # These two were creating directories named after files, let's ensure they are just directories
    os.makedirs(os.path.join(base_data_dir, "protein.sequences.v12.0.fa"), exist_ok=True)
    os.makedirs(os.path.join(base_data_dir, "uniprot_sprot.fasta"), exist_ok=True)

    db_ops = DatabaseOps(DB_CONFIG)
    create_sequences_table_sql = f"""CREATE TABLE IF NOT EXISTS sequences (id INT AUTO_INCREMENT PRIMARY KEY, identifier VARCHAR(255) NOT NULL, source VARCHAR(50), length INT, sequence LONGTEXT, description TEXT, UNIQUE KEY unique_identifier (identifier));"""
    create_uniprot_pdb_table_sql = f"""CREATE TABLE IF NOT EXISTS uniprot_pdb (id INT AUTO_INCREMENT PRIMARY KEY, source VARCHAR(255) NOT NULL, target VARCHAR(50) NOT NULL, INDEX uniprot_idx (source), INDEX pdb_idx (target), UNIQUE KEY unique_mapping (source, target));"""
    create_negative_interactions_table_sql = f"""CREATE TABLE IF NOT EXISTS negative_interactions (id INT AUTO_INCREMENT PRIMARY KEY, source VARCHAR(255), target VARCHAR(255), detection TEXT, first_author VARCHAR(255), publication VARCHAR(255), ncbi_tax_id_source VARCHAR(50), ncbi_tax_id_target VARCHAR(50), interaction_type VARCHAR(255), datasource_name VARCHAR(100), confidence VARCHAR(255));"""

    logger.info("--- Example 1: Parse FASTA and Insert to DB ---")
    if db_ops.execute_ddl(create_sequences_table_sql):
        fasta_dir_ex1 = os.path.join(base_data_dir, "protein.sequences.v12.0.fa")
        string_fasta_file = os.path.join(fasta_dir_ex1, f"dummy_protein_seqs{run_suffix}.fa")
        logger.info(f"Creating/Overwriting dummy FASTA file: {string_fasta_file}")  # Always overwrite
        with open(string_fasta_file, "w") as f:
            f.write(f">seq1{run_suffix} protein one\nACGTACGTACGTNNNN\n")
            f.write(f">seq2{run_suffix} protein two\nTTTTGGGGTTTTCCCC\n")
        fasta_records: List[Tuple[Any, ...]] = []
        batch_size = 100
        for record in tqdm(ProteinFileOps.parse_fasta_efficiently(string_fasta_file), desc="Parsing STRING FASTA"):
            fasta_records.append(
                (record['identifier'], "STRING", record['length'], record['sequence'], record['description']))
            if len(fasta_records) >= batch_size: db_ops.insert_data_batch("sequences",
                                                                          ["identifier", "source", "length", "sequence",
                                                                           "description"],
                                                                          fasta_records); fasta_records = []
        if fasta_records: db_ops.insert_data_batch("sequences",
                                                   ["identifier", "source", "length", "sequence", "description"],
                                                   fasta_records)
    logger.info("FASTA parsing and DB insertion example finished.")

    logger.info("\n--- Example 2: Parse UniProt FASTA ---")
    fasta_dir_ex2 = os.path.join(base_data_dir, "uniprot_sprot.fasta")
    uniprot_fasta_file = os.path.join(fasta_dir_ex2, f"dummy_uniprot{run_suffix}.fasta")
    logger.info(f"Creating/Overwriting dummy FASTA file: {uniprot_fasta_file}")  # Always overwrite
    with open(uniprot_fasta_file, "w") as f:
        f.write(
            f">sp|P12345{run_suffix}|TEST_HUMAN Test Protein OS=Homo sapiens\nMVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR\n")
    parsed_proteins = ProteinFileOps.parse_fasta_to_dict(uniprot_fasta_file, default_source="UNIPROT")
    if parsed_proteins:
        logger.info(f"Parsed {len(parsed_proteins)} sequences from {uniprot_fasta_file}.")
    else:
        logger.info(f"No sequences parsed from {uniprot_fasta_file}.")

    logger.info("\n--- Example 3: Parse TSV (UniProt-PDB) ---")
    if db_ops.execute_ddl(create_uniprot_pdb_table_sql):
        uniprot_pdb_tsv = os.path.join(base_data_dir, f"dummy_uniprot_pdb{run_suffix}.tsv")
        logger.info(f"Creating/Overwriting dummy TSV file: {uniprot_pdb_tsv}")  # Always overwrite
        with open(uniprot_pdb_tsv, "w") as f:
            f.write("h1\th2\n#c\n")  # Header lines
            f.write(f"UniProtID1{run_suffix}\tPDB1{run_suffix};PDB2{run_suffix}\n")
            f.write(f"UniProtID2{run_suffix}\tPDB3{run_suffix}\n")
        interactions: List[Tuple[Any, ...]] = []
        batch_size = 100
        for row in tqdm(ProteinFileOps.parse_generic_tsv(uniprot_pdb_tsv, skip_header_lines=2),
                        desc="Parsing UniProt-PDB TSV"):
            if len(row) == 2:
                uid, pids_str = row[0], row[1]
                for pid in pids_str.split(';'):
                    if uid and pid: interactions.append((uid.strip(), pid.strip()))
            if len(interactions) >= batch_size: db_ops.insert_data_batch("uniprot_pdb", ["source", "target"],
                                                                         interactions); interactions = []
        if interactions: db_ops.insert_data_batch("uniprot_pdb", ["source", "target"], interactions)
    logger.info("UniProt-PDB TSV parsing and DB insertion example finished.")

    logger.info("\n--- Example 4: Parse MITAB ---")
    if db_ops.execute_ddl(create_negative_interactions_table_sql):
        mitab_dir = os.path.join(base_data_dir, "ProteinNegativeAssociations")
        mitab_file = os.path.join(mitab_dir, f"dummy_neg_interactions{run_suffix}.mitab")
        logger.info(f"Creating/Overwriting dummy MITAB file: {mitab_file}")  # Always overwrite
        with open(mitab_file, "w") as f:
            f.write("src\ttgt\tdet\taut\tpub\ttaxSrc\ttaxTgt\tintType\tdbSrc\tconf\n")
            f.write(
                f"uniprot:P12345{run_suffix}\tuniprot:Q67890{run_suffix}\tpsi-mi:\"MI:0007\"\tAuth(24)\tpmid:123\t9606\t9606\tpsi-mi:\"MI:0407\"\tIntAct\tscore:0.8\n")
            f.write(
                f"uniprot:PABCDE{run_suffix}\tuniprot:QFGHIJ{run_suffix}\tpsi-mi:\"MI:0004\"\tSmith(23)\tpmid:456\t10090\t10090\tpsi-mi:\"MI:0915\"\tBioGRID\tconf:high\n")
        cols = ["source", "target", "detection", "first_author", "publication", "ncbi_tax_id_source",
                "ncbi_tax_id_target", "interaction_type", "datasource_name", "confidence"]
        mitab_recs: List[Tuple[Any, ...]] = []
        batch_size = 100
        for row_data in tqdm(ProteinFileOps.parse_generic_tsv(mitab_file, delimiter='\t', skip_header_lines=1),
                             desc="Parsing MITAB"):
            if len(row_data) >= len(cols): mitab_recs.append(tuple(row_data[i].strip() for i in range(len(cols))))
            if len(mitab_recs) >= batch_size: db_ops.insert_data_batch("negative_interactions", cols,
                                                                       mitab_recs); mitab_recs = []
        if mitab_recs: db_ops.insert_data_batch("negative_interactions", cols, mitab_recs)
    logger.info("MITAB parsing and DB insertion example finished.")

    logger.info("\n--- Example 5: Parse UniRef FASTA to TSV ---")
    uniref_dir = os.path.join(base_data_dir, "uniref100")
    uniref_fasta_input = os.path.join(uniref_dir, f"dummy_uniref{run_suffix}.fasta")
    uniref_tsv_output = os.path.join(uniref_dir, f"dummy_uniref_parsed{run_suffix}.tsv")
    num_dummy_uniref_records = 20000
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    logger.info(
        f"Creating/Overwriting dummy UniRef FASTA: {uniref_fasta_input} ({num_dummy_uniref_records} records).")  # Always overwrite
    with open(uniref_fasta_input, "w") as f:
        for i in range(num_dummy_uniref_records):
            seq_len = random.randint(50, 500)
            seq = "".join(random.choices(amino_acids, k=seq_len))
            f.write(
                f">UniRef100_A0A{i:06X}{run_suffix} Prot{i} n=1 Tax=Test OX=0 RepID=REP{i:06X}{run_suffix}\n{seq}\n")
    uniref_header = ["UniRef_ID", "Organism", "Sequence_Length", "Tax_ID", "Rep_ID", "Sequence"]
    try:
        with open(uniref_tsv_output, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=uniref_header, delimiter='\t')
            writer.writeheader()
            count = 0
            for rec in tqdm(ProteinFileOps.parse_uniref_fasta_to_tsv_rows(uniref_fasta_input),
                            desc="Parsing UniRef to TSV"):
                writer.writerow(rec)
                count += 1
            logger.info(f"UniRef FASTA to TSV complete. {count} records to {uniref_tsv_output}")
    except IOError as e:
        logger.error(f"Error writing UniRef TSV to '{uniref_tsv_output}': {e}")

    logger.info("\n--- Example 6: Dask Workflow ---")
    large_tsv_filepath = uniref_tsv_output
    if not os.path.exists(large_tsv_filepath) or os.path.getsize(large_tsv_filepath) < 1000:
        logger.error(f"Dask input TSV {large_tsv_filepath} missing/small. Ensure Ex5 ran.")
        return

    seq_col = "Sequence"
    dask_blocksize = '1MB'
    checkpoint_base_dir = os.path.join(base_data_dir, "dask_checkpoints")
    pq_dir1 = os.path.join(checkpoint_base_dir, f"sequences_parquet{run_suffix}")
    pq_dir2 = os.path.join(checkpoint_base_dir, f"processed_sequences_parquet{run_suffix}")

    ddf_seqs = ProteinAlgorithms.load_sequences_with_dask(large_tsv_filepath, seq_col, blocksize=dask_blocksize)

    if ddf_seqs is not None and ddf_seqs.npartitions > 0:
        logger.info(f"Dask DF from TSV head:\n{ddf_seqs.head()}")
        if ProteinFileOps.save_dask_dataframe_to_parquet(ddf_seqs, pq_dir1, overwrite=True, compression='gzip'):
            logger.info(f"Initial Dask DF saved to Parquet: {pq_dir1}")
        else:
            logger.error(f"Failed to save initial Dask DF to {pq_dir1}. Workflow cannot continue.");
            return
    else:
        logger.error("Failed to load data from TSV for Dask or Dask DF empty. Exiting.");
        return

    loaded_ddf = ProteinFileOps.load_dask_dataframe_from_parquet(pq_dir1, columns=[seq_col])
    if loaded_ddf is None or loaded_ddf.npartitions == 0:
        logger.error("Failed to load Dask DF from Parquet or it's empty. Exiting.")
        return

    meta_proc: Dict[str, Any] = {'Sequence': loaded_ddf['Sequence'].dtype, 'Sequence_Length': 'int64',
                                 'Sequence_Edges': 'object'}
    logger.info(f"Applying partition processing. Input partitions: {loaded_ddf.npartitions}, Meta: {meta_proc}")
    result_ddf = loaded_ddf.map_partitions(ProteinAlgorithms.dask_partition_processor, meta=meta_proc)

    logger.info("Computing Dask result...")
    try:
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            computed_pdf = result_ddf.compute()

        if computed_pdf.empty:
            logger.warning("Computed Dask result (computed_pdf) is empty. Skipping save to final Parquet.")
        else:
            logger.info(f"Computed Dask result head:\n{computed_pdf.head()}")
            num_out_parts = max(1, min(loaded_ddf.npartitions, (len(computed_pdf) // 10000) + 1))
            final_ddf = dd.from_pandas(computed_pdf, npartitions=num_out_parts)

            # Define the explicit Arrow schema for the processed data
            # This must match the actual structure of final_ddf columns and their order
            # Ensure the 'Sequence' dtype matches what Dask infers or what pandas produces
            # For 'object' dtype string columns, pa.string() is usually correct.
            # If loaded_ddf['Sequence'].dtype was specific like pd.StringDtype(), its .to_arrow() would be better.
            # For simplicity, assuming string for 'Sequence' column in Parquet.

            sequence_arrow_type = pa.string()  # Default to string
            # Attempt to get a more precise arrow type from the Dask series if possible
            # This is a bit more robust if the Dask series has a specific pandas extension dtype
            try:
                if hasattr(final_ddf['Sequence'].dtype, 'to_arrow'):
                    sequence_arrow_type = final_ddf['Sequence'].dtype.to_arrow()
                elif final_ddf['Sequence'].dtype == object:  # common for strings
                    sequence_arrow_type = pa.string()

            except AttributeError:  # Fallback if dtype doesn't have to_arrow
                sequence_arrow_type = pa.string()

            final_arrow_schema = pa.schema([
                pa.field('Sequence', sequence_arrow_type),
                pa.field('Sequence_Length', pa.int64()),
                pa.field('Sequence_Edges', pa.list_(
                    pa.struct([
                        pa.field('from_node', pa.string()),  # field() is good practice
                        pa.field('to_node', pa.string()),
                        pa.field('weight', pa.float64())
                    ])
                ))
            ])
            # Verify column order:
            # final_ddf_columns = final_ddf.columns.tolist()
            # schema_columns = [f.name for f in final_arrow_schema]
            # if final_ddf_columns != schema_columns:
            #    logger.error(f"Column order mismatch! DF: {final_ddf_columns}, Schema: {schema_columns}")
            #    # Potentially reorder schema or df columns here if necessary, though map_partitions meta should ensure order.

            logger.info(
                f"Saving processed Dask DF ({len(computed_pdf)} rows, {final_ddf.npartitions} parts) to Parquet: {pq_dir2} with explicit schema.")
            if ProteinFileOps.save_dask_dataframe_to_parquet(final_ddf, pq_dir2, overwrite=True, compression='gzip',
                                                             schema=final_arrow_schema):  # Pass schema
                logger.info(f"Loading 'Sequence_Edges' from final Parquet: {pq_dir2}")
                seq_edges_ddf = ProteinFileOps.load_dask_dataframe_from_parquet(pq_dir2, columns=["Sequence_Edges"])
                if seq_edges_ddf is not None:
                    logger.info(f"Sequence_Edges Dask DF head:\n{seq_edges_ddf.head()}")
                    logger.info(f"Partitions in Sequence_Edges DDF: {seq_edges_ddf.npartitions}")
            else:
                logger.error(f"Failed to save final processed Dask DF to {pq_dir2}.")
    except Exception as e:
        logger.error(f"Error during Dask computation/final saving: {e}", exc_info=True)

    logger.info("Dask workflow example finished.")
    logger.info("Main example workflow finished.")


def load_uniprot_database():
    logger.info("Starting workflow...")
    base_data_dir = "G:/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/"
    db_ops = DatabaseOps(DB_CONFIG)
    create_sequences_table_sql = f"""CREATE TABLE IF NOT EXISTS uniprot_sequences (id INT AUTO_INCREMENT PRIMARY KEY, identifier VARCHAR(255) NOT NULL, source VARCHAR(50), length INT, sequence LONGTEXT, description TEXT, UNIQUE KEY unique_identifier (identifier));"""
    logger.info("--- Parse FASTA and Insert to DB ---")
    if db_ops.execute_ddl(create_sequences_table_sql):
        fasta_dir_ex1 = os.path.join(base_data_dir, "uniprot_sprot.fasta")
        logger.info(f"Creating/Overwriting FASTA file: {fasta_dir_ex1}")
        fasta_records: List[Tuple[Any, ...]] = []
        batch_size = 100
        for record in tqdm(ProteinFileOps.parse_fasta_efficiently(fasta_dir_ex1), desc="Parsing UNIPROT FASTA"):
            fasta_records.append(
                (record['identifier'], "UNIPROT", record['length'], record['sequence'], record['description']))
            if len(fasta_records) >= batch_size: db_ops.insert_data_batch("uniprot_sequences",
                                                                          ["identifier", "source", "length", "sequence",
                                                                           "description"],
                                                                          fasta_records); fasta_records = []
        if fasta_records: db_ops.insert_data_batch("uniprot_sequences",
                                                   ["identifier", "source", "length", "sequence", "description"],
                                                   fasta_records)
    logger.info("FASTA parsing and DB insertion finished.")
