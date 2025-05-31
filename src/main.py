
# from mapper import *
# from grapher import *
from model4 import *



def process_large_tsv_in_chunks(filepath, chunk_size=10000):
    """
    Loads a large TSV file in chunks and iterates through each row.

    Args:
        filepath (str): The path to the large TSV file.
        chunk_size (int): The number of rows to read at a time.
                          Defaults to 10000.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return

    print(f"Processing file: {filepath}")
    print(f"Reading in chunks of {chunk_size} rows...")
    sequences = [] ## FIRST MEMORY FILLING UP
    try:
        # Use pd.read_csv for TSV, specifying the separator and chunksize
        # low_memory=False is often recommended when dealing with mixed data types

        for i, chunk in tqdm(enumerate(pd.read_csv(filepath, sep='\t', chunksize=chunk_size, low_memory=False))):
            # print(f"\nProcessing chunk {i+1}...")
            # Iterate through each row in the current chunk
            for index, row in chunk.iterrows():
                # --- Your processing logic for each row goes here ---
                # The 'row' variable is a pandas Series representing a single line/record.
                # You can access columns by name, e.g., row['column_name']

                # Example: print the first column of the row
                # Make sure to replace 'Column1' with an actual column name from your TSV file
                # print(f"Processing row {index}: First column value = {row.iloc[0]}")

                # Replace this pass statement with your actual row processing code

                sequences.append(row["Sequence"])
                # ----------------------------------------------------

            # print(f"Finished processing chunk {i+1}.")

        print("\nFinished processing the entire file.")

    except Exception as e:
        print(f"An error occurred: {e}")
    return sequences


# --- Testing ---
if __name__ == "__main__":
    pdb_id_to_fetch = "1A2B"  # Replace with your desired PDB ID
    save_directory = "pdb_files"  # Optional: specify a directory to save files

    # Download in PDB format
    print(f"\n--- Downloading {pdb_id_to_fetch} as .pdb ---")
    file_path_pdb, data_pdb = BioHelper.download_pdb_structure(pdb_id_to_fetch, save_directory, file_format="pdb")

    if file_path_pdb:
        print(f"PDB file saved at: {file_path_pdb}")
    # if data_pdb:
    # print("\nFirst 5 lines of PDB data:")
    # print("\n".join(data_pdb.splitlines()[:5])) # Print first 5 lines

    # Example: Download in mmCIF format
    print(f"\n--- Downloading {pdb_id_to_fetch} as .cif ---")
    pdb_id_to_fetch_cif = "4HHB"  # Using a different example for variety
    file_path_cif, data_cif = BioHelper.download_pdb_structure(pdb_id_to_fetch_cif, save_directory, file_format="cif")

    if file_path_cif:
        print(f"mmCIF file saved at: {file_path_cif}")
    # if data_cif:
    # print("\nFirst 5 lines of mmCIF data:")
    # print("\n".join(data_cif.splitlines()[:5])) # Print first 5 lines

    # Example: Invalid PDB ID
    print("\n--- Attempting to download invalid ID ---")
    download_pdb_structure("INVALID", save_directory)

    # --- Example Usage ---
    # your_large_tsv_filepath = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/uniref50_parsed_sample.tsv"
    your_large_tsv_filepath = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/uniref100_parsed.tsv"
    sequence_column = "Sequence"  # The name of the column in your TSV holding the sequences
    # batch_s = 128  # Your desired batch size for training

    parquet_checkpoint_dir_1 = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/my_dask_checkpoint_3"  # Directory for Parquet files
    parquet_checkpoint_dir_2 = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/my_dask_checkpoint_4"  # Directory for Parquet files

    # dask_sequences = None

    dask_sequences = load_sequences_with_dask(your_large_tsv_filepath, sequence_column_name=sequence_column,
                                              blocksize='16MB')

    print(dask_sequences.shape)
    print(dask_sequences.head())
    print(dask_sequences.npartitions)

    if dask_sequences is not None:
        print("Saving loaded data to Parquet checkpoint for future runs...")
        save_dask_dataframe_to_parquet(dask_sequences, parquet_checkpoint_dir_1)
    else:
        print("Failed to load data from TSV. Exiting.")
        exit()  # Or handle error appropriately

    # Try to load from Parquet checkpoint first
    print(f"Checkpoint found at {parquet_checkpoint_dir_1}. Loading from Parquet...")
    dask_sequences = load_dask_dataframe_from_parquet(
        parquet_checkpoint_dir_1,
        columns=["Sequence"]  # Specify columns you need
    )

    result_ddf = dask_sequences.map_partitions(my_partition_function,
                                               meta={'Sequence': 'object', 'Sequence_Length': 'i8',
                                                     'Sequence_Edges': 'object'})

    # Trigger computation to see the print statements and the result
    print("\nComputing result_ddf:")
    computed_result = result_ddf.compute()
    print("\nComputed result head:")
    print(computed_result.head())

    ddf_new_from_computed = dd.from_pandas(computed_result, npartitions=dask_sequences.npartitions)

    if ddf_new_from_computed is not None:
        print("Saving loaded data to Parquet checkpoint for future runs...")
        save_dask_dataframe_to_parquet(ddf_new_from_computed, parquet_checkpoint_dir_2)
    else:
        print("Failed to load data from TSV. Exiting.")
        exit()  # Or handle error appropriately

    # Try to load from Parquet checkpoint first
    print(f"Checkpoint found at {parquet_checkpoint_dir_2}. Loading from Parquet...")
    Sequence_Edges_DDF = load_dask_dataframe_from_parquet(
        parquet_checkpoint_dir_2,
        columns=["Sequence_Edges"]  # Specify columns you need
    )

    print(Sequence_Edges_DDF.shape)
    print(Sequence_Edges_DDF.head())
    print(Sequence_Edges_DDF.npartitions)

    # # Example usage:
    # filepath = "seqs-2.txt"  # Replace with your FASTA file path
    # parsed_data = parse_fasta(filepath)

    # tsv_file_path = 'G:/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/uniref50_parsed.tsv'
    # tsv_file_path = 'G:/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/uniref50_parsed_sample.tsv'
    # tsv_file_path = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/uniref100_parsed.tsv"
    tsv_file_path = "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/uniref50_parsed_sample.tsv"
    custom_chunk_size = 300000  # Example: process 20,000 rows at a time
    sequences = process_large_tsv_in_chunks(tsv_file_path, chunk_size=custom_chunk_size)

    if sequences:
        all_sequences = []
        counter = 0
        for document in tqdm(sequences):
            edges = []
            unique_character_vocab = []
            character_vocab = {}
            inverse_character_vocab = {}
            character_edges = []
            # token_edges = []
            # print("Description:", description)
            # print("Sequence:", document)
            # print("-" * 20)  # Separator between sequences
            main_body = Algorithms.strip_non_alphanumeric(document)
            attributes = [main_body]
            final_character_walk = []
            # final_character_walk.append(chr(27))
            for attribute in attributes:
                # tokens = re.sub(" +", " ", attribute.rstrip().lstrip()).lower().split(" ")
                for token in attribute:
                    for c in token:
                        final_character_walk.append(c)
                    # if token != tokens[-1]:
                    #     final_character_walk.append(chr(32))
            #     final_character_walk.append(chr(31))
            # final_character_walk.append(chr(28))
            unique_character_vocab = unique_character_vocab + final_character_walk
            final_character_walk = Algorithms.encode_characters(final_character_walk)
            for i in range(len(final_character_walk)):
                if i + 1 < len(final_character_walk):
                    c = final_character_walk[i]
                    next_c = final_character_walk[i + 1]
                    edges.append(str(c) + "|" + str(next_c))
            unique_character_vocab = list(set(unique_character_vocab))
            edge_counter = dict(Counter(edges))
            for c in unique_character_vocab:
                character_vocab[c] = [str(ord(c))]
            inverse_character_vocab = Algorithms.invert_dict(character_vocab)
            for k, v in edge_counter.items():
                ns2 = k.split("|")
                character_edges.append((str(ns2[0]), str(ns2[1]), float(v)))
            all_sequences.append(character_edges)
            counter += 1
        # print(all_sequences)
        # print(len(all_sequences))
        # print(str(counter))

        Algorithms.save_checkpoint(all_sequences, "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/all_sequences.pkl")

        raw_edges = []
        node_labels = []
        for sequence in all_sequences:
            last_node = None
            for e in sequence:
                raw_edges.append((str(e[0]), str(e[1]), float(1), float(1)))
                node_labels.append(str(e[0]))
                node_labels.append(str(e[1]))
                last_node = str(e[1])
            raw_edges.append((last_node, "32", float(1), float(1)))
            node_labels.append("32")


        node_labels = list(set(node_labels))

        nodes = {}
        # for nl in node_labels:
        #     nodes[nl] = nl


        graph = DirectedGraph(nodes=nodes, edges=raw_edges)
        del raw_edges[:]
        graph.node_labels(True)
        graph.node_features(None, 64)
        print(graph)
        print("Number of Classes: " + str(graph.number_of_classes))
        # print(graph.edges)

        Algorithms.save_checkpoint(graph, "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/graph.pkl")



        # Model 4 Vanilla DiGCN
        hp4 = Hyperparameters()
        hp4.add("trials", 1)
        hp4.add("K", 2)
        hp4.add("epochs", 200)
        hp4.add("patience", 200)
        hp4.add("split", 70)
        hp4.add("batch_size", graph.number_of_nodes)
        hp4.add("regularization_rate", 0.0005)
        hp4.add("learning_rate", 0.001)
        hp4.add("dropout_rate", 0.5)
        hp4.add('input_layer_dimension', graph.dimensions)
        hp4.add('layer_1_dimension', 1024)
        hp4.add('layer_2_dimension', 1024)
        hp4.add('layer_3_dimension', 1024)
        hp4.add('output_layer_dimension', graph.number_of_classes)

        m4 = Model4(hp4, graph)
        all_embeddings = m4.run()

        # Create a sample list object

        # Define the filename


        Algorithms.save_checkpoint(all_embeddings, "C:/Users/islam/My Drive/Knowledge/Research/TWU/Topics/Data Mining in Proteomics/Projects/Link Prediction in Protein Interaction Networks via Sequence Embedding/Data/all_embeddings.pkl")