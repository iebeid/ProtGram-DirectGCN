<h1 align="center">ProtGram-DirectGCN</h1>

<p align="center">
  A deep learning pipeline for protein-protein interaction (PPI) prediction using a novel combination of n-gram graphs and Graph Convolutional Networks (GCNs).
</p>

---

## Overview

This project introduces ProtGram-DirectGCN, a method to generate powerful protein sequence embeddings for predicting interactions. The core pipeline involves:
1.  **ProtGram: Graph Construction**: Building directed n-gram graphs from protein sequences, where nodes are amino acid n-grams and edges represent their co-occurrence.
2.  **DirectGCN: Graph Convolutional Neural Network Training**: Training a Custom Directed Graph Convolutional Network on these graphs to learn topologically-aware embeddings.
3.  **PPI Prediction**: Relying on the learned embeddings to train a downstream classifier for link prediction (PPI).
4.  **Benchmarking**: Evaluating the model against other GNNs and established protein embedding techniques.

## Features

- **Scalable Graph Construction**: Utilizes Dask to build massive n-gram graphs from large protein sequence files (e.g., UniRef50) without memory crashes.
- **Novel GCN Architecture**: Implements `DirectGCN`, a custom GCN that leverages directed graph information and hierarchical gating for enhanced representation learning.
- **Comprehensive Benchmarking**: Includes a full suite for comparing against standard GNNs (GCN, GAT, GraphSAGE, etc.) and baseline embedding methods (Word2Vec, Transformers, LSTM).
- **Automated Environment Setup**: A robust setup script that creates a self-contained Conda environment with all necessary dependencies, including the correct CUDA toolkit and cuDNN versions.
- **Reproducibility**: End-to-end pipeline with fixed random seeds and deterministic algorithms to ensure run-to-run consistency.
- **Experiment Tracking**: Integrated with MLflow to log parameters, metrics, and artifacts for all experiments.

---

## Installation (One-Time Setup)

The setup process has been streamlined into a single script. Follow these steps for a one-time setup of the project and its environment.

> [!NOTE]
> This project is designed to run in a Linux environment. For Windows users, it is **strongly recommended** to use the **Windows Subsystem for Linux (WSL) 2**.

### Prerequisites

Before you begin, ensure you have the following installed:

1.  **WSL 2**: If on Windows, ensure you have a WSL 2 distribution (e.g., Ubuntu) installed and running.
2.  **Anaconda or Miniconda**: The Conda package manager must be installed **inside your WSL 2 instance**.
3.  **(Optional but Recommended) NVIDIA GPU**: For significantly faster training, an NVIDIA GPU is highly recommended.
    -   **Driver Installation**: You only need to install the latest **NVIDIA Game Ready or Studio drivers** on your main Windows system.
    -   **No CUDA Toolkit on Host**: You do **not** need to manually install the NVIDIA CUDA Toolkit or cuDNN on Windows or within WSL. The setup script will handle the installation of all necessary CUDA libraries inside the self-contained Conda environment.

### One-Step Installation

1.  **Download the Setup Script**:
    -   Navigate to the project's GitHub repository in your web browser.
    -   Go to the file: `configuration/reset.sh`.
    -   Download this single script file.

2.  **Place and Run the Script**:
    -   Move the downloaded `reset.sh` file into your WSL home directory (`~`). You can access this from Windows File Explorer at `\\wsl$\<Your_Distro_Name>\home\<your_username>\`.
    -   Open your WSL terminal. You should be in your home directory and in the `(base)` Conda environment.
    -   Make the script executable and run it:

    ```sh
    # From your WSL home directory (~)
    chmod +x reset.sh
    bash reset.sh
    ```

This single script will perform all necessary actions:
-   Create the `ppi-env` Conda environment with the correct Python version.
-   Clone the full `ProtGram-DirectGCN` repository into `~/documents/projects/`.
-   Install all required dependencies, including the correct versions of PyTorch, TensorFlow, and the CUDA toolkit, into the `ppi-env`.

The process may take several minutes. Once complete, your environment and project are fully set up.

> [!WARNING]
> **Do Not Manually Edit `.bashrc` or `.zshrc`**
>
> This project is designed to be **self-contained**. The setup script installs all necessary components, including the CUDA Toolkit and cuDNN, directly into the Conda environment. You **should not** manually `export` system-wide CUDA paths in your shell configuration files, as this can cause conflicts with the environment's managed libraries.

---

## How to Run

After the one-time installation with `reset.sh` is complete, all subsequent runs are managed with the `start.sh` script inside the project directory.

### Step 1: Navigate to the Project Directory

Open a new WSL terminal. The `ppi-env` is **not** activated by default. The `start.sh` script will handle activation.

```sh
cd ~/documents/projects/ProtGram-DirectGCN
```

### Step 2 (Optional): Configure Your Run

Before running, you can edit `configuration/config.py` to enable or disable different parts of the pipeline. See the **Configuration Guide** below for details.

### Step 3: Execute the Pipeline

Run the `start.sh` script. It will automatically activate the correct Conda environment, update the repository, and start the main application.

```sh
# From the project root directory (e.g., ~/documents/projects/ProtGram-DirectGCN)
bash start.sh
```

### Example 2: Run Only the Integrated Tests

To verify the environment and run all unit and smoke tests without executing the long-running pipelines, edit `configuration/config.py` as follows:

```python
# In configuration/config.py

# Enable only the tests
self.RUN_INTEGRATED_TESTS = True

# Disable all other major pipelines
self.RUN_GCN_PIPELINE = False
self.RUN_MAIN_PPI_EVALUATION = False
self.RUN_BENCHMARKING_PIPELINE = False
# ... and so on for other pipeline flags
```

Then, run the `start.sh` script.

### Example 3: Run Only the Main PPI Evaluation

To run only the final PPI evaluation on embeddings that have already been generated, edit `configuration/config.py`:

```python
# In configuration/config.py

# Disable all embedding generation pipelines
self.RUN_GCN_PIPELINE = False
self.RUN_LSTM_PIPELINE = False
# ... etc.

# Enable the final evaluation
self.RUN_MAIN_PPI_EVALUATION = True
```

Then, run the `start.sh` script.

---

## Configuration Guide (`configuration/config.py`)

The pipeline's behavior is controlled by the `configuration/config.py` file. Below are the key parameters you can adjust, grouped by function.

### 1. Pipeline Control Flags
These boolean flags enable or disable major components of the pipeline. This is useful for running only specific experiments.

-   `RUN_GCN_PIPELINE`: (`True`/`False`) Runs the full ProtGram-DirectGCN embedding generation pipeline.
-   `RUN_LSTM_PIPELINE`, `RUN_WORD2VEC_PIPELINE`, `RUN_TRANSFORMER_PIPELINE`: (`True`/`False`) Flags to run baseline embedding generation pipelines.
-   `RUN_BENCHMARKING_PIPELINE`: (`True`/`False`) Runs the GNN benchmarking suite on standard node classification datasets (Cora, PubMed, etc.).
-   `RUN_NETWORK_EMBEDDING_BENCHMARKING`: (`True`/`False`) Runs the network embedding benchmarking suite (Node2Vec, etc.).
-   `RUN_MAIN_PPI_EVALUATION`: (`True`/`False`) Runs the final protein-protein interaction (PPI) prediction evaluation using all specified embeddings.
-   `RUN_INTEGRATED_TESTS`: (`True`/`False`) Runs a suite of verification and smoke tests on startup to ensure the environment is configured correctly.
-   `RUN_SINGLETON_GCN_EVAL`: (`True`/`False`) Runs a fast evaluation of multiple GNNs on the n=1 graph for rapid prototyping and preliminary analysis.
-   `SEQUENCE_DOWNSAMPLE_FRACTION`: (`float` or `None`) A float between `0.0` and `1.0`. If set, a random fraction of sequences from the input FASTA file will be used. Set to `None` or `1.0` to disable.
-   `ENABLE_FILE_LOGGING`: (`True`/`False`) If `True`, all console output is saved to a timestamped log file in the `results/logs` directory.

### 2. Data Sources and Paths
This section controls the automatic download of data and defines key file paths.

-   `DATA_SOURCES`: A dictionary defining the URLs and destination paths for required data. The script automatically handles downloads from standard FTP sites and Google Drive.
-   `PERSISTENT_DATA_CACHE`: Defines a path (e.g., `~/.cache/protgram_directgcn`) where large, reusable files like UniRef50 or ProtT5 embeddings are stored to avoid re-downloading on subsequent project resets.

### 3. ProtGram-DirectGCN Pipeline Parameters
These parameters control the core methodology of the project.

#### Graph Building
-   `USE_FAST_GRAPH_BUILDER`: (`True`/`False`) If `True` (default), uses the new, scalable Dask-based graph builder. If `False`, uses the legacy builder.
-   `PROTGRAM_NGRAM_MAX_N`: (`int`) The maximum n-gram size to build graphs for (e.g., `3` builds graphs for n=1, n=2, and n=3).
-   `GRAPH_BUILDER_WORKERS`: (`int` or `None`) The number of parallel workers to use for building graphs. Defaults to most of your CPU cores.

#### ID Mapping
-   `ID_MAPPING_MODE`: (`'regex'`, `'file'`, `'api'`, `'none'`)
    -   `'regex'`: (Default) A fast method that extracts UniProt IDs directly from FASTA headers. Works well for standard UniProt/UniRef formats.
    -   `'file'`: Uses a memory-efficient method to parse the large `idmapping.dat` file and create a fast, on-disk lookup database. More robust for non-standard headers.
    -   `'api'`: Queries the live UniProt API to map IDs. Very slow and subject to rate limits.
    -   `'none'`: Performs no ID mapping.

#### Model Architecture & Training
-   `PROTGRAM_MODELS_TO_TRAIN`: (`list[str]`) A list of GCN variants to train on the ProtGram graphs. Options include `'directgcn'`, `'gcn'`, `'gat'`, etc.
-   `DIRECTGCN_HIDDEN_LAYER_DIMS`: (`list[int]`) A list defining the number of units in each hidden layer of the main `DirectGCN` model.
-   `PROTGRAM_EPOCHS_PER_LEVEL`: (`int`) Number of training epochs for each n-gram level.
-   `PROTGRAM_LR`: (`float`) The learning rate for the GCN optimizer.
-   `PROTGRAM_USE_LR_SCHEDULER`: (`True`/`False`) Enables a learning rate scheduler that reduces the LR on a plateau.
-   `PROTGRAM_USE_EARLY_STOPPING`: (`True`/`False`) Enables early stopping to prevent overfitting.

#### Self-Supervised Tasks & Pooling
-   `PROTGRAM_TASK_TYPES_PER_LEVEL`: (`dict`) Maps each n-gram level (`int`) to a self-supervised task (`str`). Options: `'community'`, `'next_node'`, `'masked_node'`.
-   `PROTGRAM_PROTEIN_POOLING_STRATEGY`: (`'attention'`, `'mean'`, `'max'`, `'sum'`) Method for pooling n-gram embeddings to create a final protein embedding.
-   `PROTGRAM_HIERARCHICAL_POOLING_STRATEGY`: (`'attention'`, `'mean'`) Method for pooling `(n-1)`-gram embeddings to initialize features for the `n`-gram graph.

#### Cluster-GCN Strategy
-   `PROTGRAM_USE_CLUSTER_TRAINING`: (`True`/`False`) If `True`, uses the Cluster-GCN strategy for very large graphs to make training feasible on a single GPU.
-   `PROTGRAM_TARGET_NODES_PER_CLUSTER`: (`int`) The desired number of nodes in each graph cluster/subgraph.

### 4. Baseline Embedding Pipeline Parameters
These sections control the alternative embedding generation methods used for comparison.

-   **Word2Vec (`_setup_word2vec_params`)**:
    -   `W2V_VECTOR_SIZE`: (`int`) The dimensionality of the Word2Vec embeddings.
    -   `W2V_WINDOW`: (`int`) The context window size.
-   **Transformers (`_setup_transformer_params`)**:
    -   `TRANSFORMER_MODELS_TO_RUN`: (`list[dict]`) A list of models from Hugging Face to run (e.g., `Rostlab/prot_bert`).
    -   `TRANSFORMER_BASE_BATCH_SIZE`: (`int`) The base batch size for inference, which is scaled by a model-specific multiplier.
-   **LSTM (`_setup_lstm_params`)**:
    -   `LSTM_EMBEDDING_DIM`: (`int`) The dimensionality of the initial token embeddings.
    -   `LSTM_HIDDEN_DIM`: (`int`) The number of units in the LSTM hidden layers.

### 5. PPI Evaluation Parameters
This section configures the final link prediction task.

-   `EVAL_EDGE_EMBEDDING_METHOD`: (`'concatenate'`, `'hadamard'`, etc.) The method used to combine the embeddings of two proteins to create an edge embedding.
-   `EVAL_N_FOLDS`: (`int`) The number of folds for cross-validation.
-   `EVAL_MLP_DENSE1_UNITS`, `EVAL_MLP_DROPOUT1_RATE`, etc.: Parameters defining the architecture of the downstream Multi-Layer Perceptron (MLP) classifier.
-   `LP_EXTERNAL_EMBEDDINGS_TO_EVALUATE`: A list of dictionaries pointing to pre-existing embedding files (like ProtT5) to include in the final comparison.

### 6. Benchmarking & Singleton Evaluation Parameters

-   **GNN Benchmarking**:
    -   `BENCHMARK_NODE_CLASSIFICATION_DATASETS`: (`list[str]`) A list of standard datasets (`'Cora'`, `'CiteSeer'`, etc.) to use for the GNN benchmarking suite.
    -   `BENCHMARK_GNN_MODELS_TO_RUN`: (`list[str]`) The GNN models to test in the benchmark.
-   **Singleton Evaluation**:
    -   `SINGLETON_EVAL_MODELS_TO_RUN`: (`list[str]`) The GNN models to test on the n=1 ProtGram graph.
    -   `SINGLETON_EVAL_EPOCHS`: (`int`) Number of epochs for this rapid evaluation.

### 7. MLflow Parameters

-   `USE_MLFLOW`: (`True`/`False`) Enables or disables experiment tracking with MLflow.
-   `MLFLOW_TRACKING_URI`: The location to store MLflow runs. Defaults to `results/mlruns`.

---

## Project Structure

-   `configuration/`: Contains all configuration files, including `config.py` and the environment setup scripts (`setup.py`, `reset.sh`).
-   `source/`: The main source code for the project.
    -   `data_builders/`: Logic for constructing graphs from sequence data.
    -   `trainers/`: Classes that orchestrate the training of different models (GCN, LSTM, etc.).
    -   `models/`: Definitions of the neural network architectures (GCNs, MLP, etc.).
    -   `experiments/`: High-level experiment pipelines, like the final PPI evaluation.
    -   `benchmarkers/`: Code for running benchmarks against standard datasets.
    -   `testers/`: Unit, smoke, and verification tests.
    -   `utils/`: Helper utilities for data processing, logging, and results reporting.
-   `results/`: The default output directory for all generated files.
    -   `graph_objects/`: Stores the serialized n-gram graph files (`.pkl`).
    -   `gcn_embeddings/`: Stores the final protein embeddings (`.h5`) generated by the ProtGram-GCN pipeline.
    -   `evaluation_results/`: Contains all plots, tables, and summaries from the PPI evaluation.
    -   `logs/`: Contains timestamped log files of all console output.
    -   `mlruns/`: The directory for MLflow experiment tracking data.
-   `data/`: The default directory for all input data.
    -   `sequences/`: Input protein FASTA files.
    -   `ground_truth/`: Positive and negative protein interaction pair lists.
    -   `models/`: Pre-trained models, such as ProtT5.

---

## Citing

The article associated with this repository is currently under review.

## License

This project is licensed under the MIT License.
