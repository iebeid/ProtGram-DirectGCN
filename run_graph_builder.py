# run_graph_builder.py
import time
import os
import shutil
import tempfile
from pathlib import Path
import dask  # Import dask for config setting
import pandas as pd
import pyarrow.parquet as pq
import networkx as nx
import community as community_louvain

from config import Config
from src.pipeline.data_builder import GraphBuilder
from src.utils.data_utils import DataUtils
from src.utils.graph_utils import DirectedNgramGraph # Import for instantiation

if __name__ == "__main__":
    script_start_time = time.time()
    DataUtils.print_header("Starting GraphBuilder Intermediate and Object Test (Synchronous Dask)")

    base_test_dir = tempfile.mkdtemp()
    temp_input_dir = os.path.join(base_test_dir, "input")
    os.makedirs(temp_input_dir, exist_ok=True)

    fasta_content = ">seq1\nACGT\n>seq2\nTTAC\n>seq3\nAGA\n"
    fasta_path = os.path.join(temp_input_dir, "test_sequences.fasta")
    with open(fasta_path, "w") as f:
        f.write(fasta_content)

    intermediate_temp_dir = os.path.join(base_test_dir, "intermediate_output")
    os.makedirs(intermediate_temp_dir, exist_ok=True)

    # This directory is not strictly used by this script for final output,
    # but config needs it.
    final_graph_objects_dir = os.path.join(base_test_dir, "final_graphs_test_script")
    os.makedirs(final_graph_objects_dir, exist_ok=True)


    config = Config()
    config.GCN_INPUT_FASTA_PATH = Path(fasta_path)
    config.GRAPH_OBJECTS_DIR = Path(final_graph_objects_dir)
    config.BASE_OUTPUT_DIR = Path(base_test_dir)
    config.DEBUG_VERBOSE = True # Ensure debug prints are active

    n_value_to_test = 1
    num_dask_partitions_for_test = 1

    print(f"--- Phase 1: Testing _create_intermediate_files for n={n_value_to_test} ---")
    print(f"  Input FASTA: {fasta_path}")
    print(f"  Intermediate output to: {intermediate_temp_dir}")

    dask.config.set(scheduler='synchronous')
    print("  Dask scheduler set to 'synchronous'.")

    original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("  Temporarily set CUDA_VISIBLE_DEVICES=-1 for the main process.")

    intermediate_files_created = False
    try:
        GraphBuilder._create_intermediate_files(
            n_value_to_test,
            intermediate_temp_dir,
            str(config.GCN_INPUT_FASTA_PATH),
            num_dask_partitions_for_test
        )
        print(f"--- Phase 1 (_create_intermediate_files for n={n_value_to_test}) COMPLETED ---")

        expected_map_file = os.path.join(intermediate_temp_dir, f'ngram_map_n{n_value_to_test}.parquet')
        expected_edge_file = os.path.join(intermediate_temp_dir, f'edge_list_n{n_value_to_test}.txt')

        if os.path.exists(expected_map_file) and os.path.getsize(expected_map_file) > 0:
            print(f"  OK: Ngram map file created and not empty: {expected_map_file}")
            intermediate_files_created = True
        else:
            print(f"  FAIL: Ngram map file NOT created or is empty: {expected_map_file}")

        if os.path.exists(expected_edge_file): # Edge file can be empty if no edges
            print(f"  OK: Edge list file created: {expected_edge_file} (size: {os.path.getsize(expected_edge_file)})")
        else:
            print(f"  FAIL: Edge list file NOT created: {expected_edge_file}")
            intermediate_files_created = False # If map exists but edges don't, still an issue

    except Exception as e:
        print(f"--- Phase 1 (_create_intermediate_files) FAILED: {e} ---")
        import traceback
        traceback.print_exc()
        intermediate_files_created = False

    if intermediate_files_created:
        print(f"\n--- Phase 2: Testing Graph Object Creation and Stats for n={n_value_to_test} ---")
        try:
            # Load n-gram map
            map_table_read = pq.read_table(expected_map_file, columns=['id', 'ngram'])
            nodes_df = map_table_read.to_pandas()
            idx_to_node = nodes_df.set_index('id')['ngram'].to_dict()
            print(f"  Loaded {len(idx_to_node)} n-grams from map file.")

            # Load edge list
            weighted_edge_list_tuples = []
            if os.path.exists(expected_edge_file) and os.path.getsize(expected_edge_file) > 0:
                edge_df = pd.read_csv(expected_edge_file, sep=' ', header=None, names=['source', 'target'], dtype=int)
                print(f"  Loaded {len(edge_df)} raw edges from edge list file.")
                if not edge_df.empty:
                    weighted_edge_df = edge_df.groupby(['source', 'target']).size().reset_index(name='weight')
                    print(f"  [DEBUG] Weighted edge_df for n={n_value_to_test} (from test script):")
                    print(weighted_edge_df.head())
                    print(f"  [DEBUG] Number of unique edges in weighted_edge_df: {len(weighted_edge_df)}")
                    weighted_edge_list_tuples = [tuple(x) for x in weighted_edge_df[['source', 'target', 'weight']].to_numpy()]
                    print(f"  [DEBUG] weighted_edge_list_tuples (sample): {weighted_edge_list_tuples[:5] if weighted_edge_list_tuples else 'Empty'}")
                    print(f"  Aggregated {len(edge_df)} raw transitions into {len(weighted_edge_df)} unique weighted edges.")
                else:
                    print("  Edge file was empty, no weighted edges to aggregate.")
            else:
                print("  Edge file is empty or does not exist. No edges will be processed.")

            # Instantiate DirectedNgramGraph
            print(f"  Instantiating DirectedNgramGraph object for n={n_value_to_test}...")
            graph_object = DirectedNgramGraph(
                nodes=idx_to_node,
                edges=weighted_edge_list_tuples,
                epsilon_propagation=config.GCN_PROPAGATION_EPSILON
            )
            graph_object.n_value = n_value_to_test # Manually set for context if needed by stats
            print("  DirectedNgramGraph object instantiated.")

            # Print Graph Statistics
            print(f"    --- Graph Statistics for n={n_value_to_test} (from test script) ---")
            num_nodes = graph_object.number_of_nodes
            num_edges = graph_object.number_of_edges
            print(f"      Nodes: {num_nodes}")
            print(f"      Edges (unique weighted from graph_object.number_of_edges): {num_edges}")

            if num_nodes > 1:
                possible_edges_no_self_loops = num_nodes * (num_nodes - 1)
                density = num_edges / possible_edges_no_self_loops if possible_edges_no_self_loops > 0 else 0
                print(f"      Density (E / N(N-1)): {density:.4f}")
            else:
                print("      Density: N/A (graph has <= 1 node)")

            is_complete = False
            if num_nodes > 1 and possible_edges_no_self_loops > 0:
                if num_edges == possible_edges_no_self_loops:
                    is_complete = True
            elif num_nodes == 1 and num_edges == 0:
                is_complete = True
            print(f"      Is Complete (all N*(N-1) directed edges present): {is_complete}")

            if num_nodes > 0:
                G_nx = nx.DiGraph()
                G_nx.add_nodes_from(range(num_nodes))
                edges_for_nx = [(u, v, {'weight': w}) for u, v, w in graph_object.edges]
                G_nx.add_edges_from(edges_for_nx)

                num_wcc = nx.number_weakly_connected_components(G_nx)
                print(f"      Weakly Connected Components: {num_wcc}")

                if num_edges > 0:
                    num_scc = nx.number_strongly_connected_components(G_nx)
                    print(f"      Strongly Connected Components: {num_scc}")
                    G_undirected_nx = G_nx.to_undirected()
                    if G_undirected_nx.number_of_edges() > 0:
                        partition = community_louvain.best_partition(G_undirected_nx, random_state=config.RANDOM_STATE)
                        num_communities = len(set(partition.values()))
                        print(f"      Louvain Communities (on undirected graph): {num_communities}")
                    else:
                        print("      Louvain Communities: N/A (no edges in undirected version)")
                else:
                    print("      Strongly Connected Components: N/A (no edges)")
                    print("      Louvain Communities: N/A (no edges)")

                avg_in_degree = num_edges / num_nodes if num_nodes > 0 else 0.0
                avg_out_degree = num_edges / num_nodes if num_nodes > 0 else 0.0
                print(f"      Average In-Degree: {avg_in_degree:.2f}")
                print(f"      Average Out-Degree: {avg_out_degree:.2f}")

                if num_nodes > 1:
                    in_degree_centrality = nx.in_degree_centrality(G_nx)
                    avg_in_degree_centrality = sum(in_degree_centrality.values()) / num_nodes
                    print(f"      Avg. In-Degree Centrality (normalized): {avg_in_degree_centrality:.4f}")
                    out_degree_centrality = nx.out_degree_centrality(G_nx)
                    avg_out_degree_centrality = sum(out_degree_centrality.values()) / num_nodes
                    print(f"      Avg. Out-Degree Centrality (normalized): {avg_out_degree_centrality:.4f}")
                else:
                    print("      Avg. Degree Centrality: N/A (graph has <= 1 node)")
            print(f"    --- End of Graph Statistics for n={n_value_to_test} (from test script) ---\n")
            print(f"--- Phase 2 (Graph Object Creation Test for n={n_value_to_test}) COMPLETED ---")

        except Exception as e:
            print(f"--- Phase 2 (Graph Object Creation Test) FAILED: {e} ---")
            import traceback
            traceback.print_exc()

    if original_cuda_visible is None:
        if "CUDA_VISIBLE_DEVICES" in os.environ: del os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
    print("  Restored original CUDA_VISIBLE_DEVICES setting for the main process.")

    if os.path.exists(base_test_dir):
        print(f"Cleaning up temporary directory: {base_test_dir}")
        shutil.rmtree(base_test_dir)

    print(f"Total time for script: {time.time() - script_start_time:.2f}s")
