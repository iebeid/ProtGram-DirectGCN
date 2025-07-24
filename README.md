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

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Anaconda or Miniconda**: This project uses the Conda package manager to handle its complex dependencies. You can download it from the [official Anaconda website](https://www.anaconda.com/products/distribution).
2.  **(Optional but Recommended) NVIDIA GPU and Drivers**: For significantly faster training, an NVIDIA GPU is recommended. Make sure you have the [latest NVIDIA drivers](https://www.nvidia.com/download/index.aspx) installed for your system. The setup script will install the necessary CUDA toolkit and libraries inside the Conda environment.

> [!IMPORTANT]
> **Important Note for GPU Users on Windows**
> If you are using an NVIDIA GPU on Windows, you only need to install the **NVIDIA Game Ready or Studio drivers** on your main Windows system. You do **not** need to manually download and install the NVIDIA CUDA Toolkit or cuDNN on Windows itself. The WSL 2 environment (recommended below) will use your Windows drivers to access the GPU, and the `setup.py` script will correctly install the required CUDA and cuDNN libraries inside your Linux Conda environment.

## Installation Guide

The setup process is designed to be straightforward. Follow these four steps to create the environment and install all required packages.

> **Note for Windows Users**: If you are running on Windows, it is strongly recommended to use the [Windows Subsystem for Linux (WSL) 2](https://learn.microsoft.com/en-us/windows/wsl/install) to create a Linux environment. The following commands should be run inside a WSL terminal. If you are on a Linux-based OS like Ubuntu, you can proceed directly.

### Step 1: Create a Conda Environment

First, open your terminal (or Anaconda Prompt on Windows) and create a new Conda environment. We recommend using Python 3.11. This command will create an environment named `ppi-env`.

```sh
conda create -n ppi-env python=3.11 -y
```

### Step 2: Activate the Environment

Before installing the project dependencies, you **must** activate the environment you just created. All subsequent commands in your terminal session should be run from within this activated environment.

```sh
conda activate ppi-env
```

You will know the environment is active because its name, `(ppi-env)`, will appear at the beginning of your terminal prompt.

### Step 3: Clone the Repository

With your environment active, clone this repository to your local machine using the following command.
**Important**: After cloning, you must check out the `v2` branch, which contains the stable version of the project.

```sh
git clone https://github.com/iebeid/ProtGram-DirectGCN.git
cd ProtGram-DirectGCN
git checkout v2
```

### Step 4: Run the Installation Script

Now that you are inside the project directory, navigate to the `configuration` folder and run the `setup.py` script.

```sh
# Navigate to the configuration directory from the project root
cd configuration

# Run the setup script
python setup.py
```

This script will automate the entire installation process within your active `ppi-env`. It will:

- Install the correct versions of the CUDA toolkit and cuDNN.
- Install TensorFlow and PyTorch with GPU support.
- Install all other required data science and bioinformatics libraries like `scikit-learn`, `pandas`, and `torch-geometric`.

The process may take several minutes as it downloads and installs many large packages.

### A Note on Environment Variables and Paths

> [!WARNING]
> **Do Not Manually Edit `.bashrc` or `.zshrc`**
>
> This project is designed to be **self-contained**. The `setup.py` script installs all necessary components, including the CUDA Toolkit, cuDNN, and compilers, directly into the Conda environment (e.g., into the `.../ppi-env/` folder).
>
> You **should not** add lines like `export CUDA_HOME=/usr/local/cuda-12.5` to your `.bashrc` file. This is because:
>
> 1.  **It Points to the Wrong Location**: The `export` commands you mentioned refer to a system-wide CUDA installation in `/usr/local/`. This project uses the CUDA toolkit installed and managed by Conda inside the `ppi-env`.
> 2.  **It Can Cause Conflicts**: Manually setting these paths can override the Conda environment's settings, forcing the project to use a conflicting or non-existent CUDA version, which will lead to compilation or runtime errors.
>
> The correct paths are handled for you automatically when you run `conda activate ppi-env` and when the `main.py` script executes. If you encounter "command not found" errors, the first step should always be to ensure your Conda environment is activated correctly.

---

## Configuration Guide (`configuration/config.py`)

Before running the main pipeline via `python main.py`, you can customize the pipeline's behavior by editing the `configuration/config.py` file. This guide explains the key parameters you can adjust.

### 1. Pipeline Control Flags
These boolean flags allow you to enable or disable major components of the pipeline. This is useful for running only specific experiments.

- `RUN_GCN_PIPELINE`: (`True`/`False`) Runs the full ProtGram-DirectGCN embedding generation pipeline.
- `RUN_LSTM_PIPELINE`: (`True`/`False`) Runs the baseline LSTM embedding generation pipeline.
- `RUN_WORD2VEC_PIPELINE`: (`True`/`False`) Runs the baseline Word2Vec embedding generation pipeline.
- `RUN_TRANSFORMER_PIPELINE`: (`True`/`False`) Runs the baseline Transformer (e.g., ProtBERT) embedding generation pipeline.
- `RUN_BENCHMARKING_PIPELINE`: (`True`/`False`) Runs the GNN benchmarking suite on standard node classification datasets (Cora, PubMed, etc.).
- `RUN_NETWORK_EMBEDDING_BENCHMARKING`: (`True`/`False`) Runs the network embedding benchmarking suite (Node2Vec, DeepWalk) on the same datasets.
- `RUN_MAIN_PPI_EVALUATION`: (`True`/`False`) Runs the final protein-protein interaction (PPI) prediction evaluation using the generated embeddings.
- `RUN_INTEGRATED_TESTS`: (`True`/`False`) Runs a suite of verification and smoke tests on startup to ensure the environment is configured correctly. Highly recommended to keep `True`.
- `RUN_DUMMY_TEST`: (`True`/`False`) Runs a quick, small-scale version of the PPI evaluation on dummy data to verify the evaluation pipeline is working.
- `SEQUENCE_DOWNSAMPLE_FRACTION`: (`float` or `None`) A float between `0.0` and `1.0`. If set, a random fraction of sequences from the input FASTA file will be used for the main experiments. Set to `None` or `1.0` to disable downsampling.
- `ENABLE_FILE_LOGGING`: (`True`/`False`) If `True`, all console output is saved to a timestamped log file in the `results/logs` directory.

### 2. Data Sources and Paths
This section controls the automatic download of data and defines key file paths.

- `DATA_SOURCES`: A dictionary defining the URLs and destination paths for required data.
    - **Action Required**: You **must** replace the placeholder URLs for `POS_INTERACTIONS` and `NEG_INTERACTIONS` with the actual URLs to your positive and negative interaction data files.
- `SEQUENCE_FILE_PATHS`: A list of `Path` objects pointing to the FASTA files you wish to process. The pipeline will iterate through each file, running a separate set of experiments for each one.
- Other path variables are generated automatically based on `PROJECT_ROOT` and should not need to be changed.

### 3. ProtGram-DirectGCN Pipeline Parameters
These parameters control the core methodology of the project.

#### Graph Building
- `GCN_NGRAM_MAX_N`: (`int`) The maximum n-gram size to build graphs for (e.g., `3` will build graphs for n=1, n=2, and n=3).
- `GRAPH_BUILDER_WORKERS`: (`int` or `None`) The number of parallel workers to use for building graphs. Defaults to most of your CPU cores.

#### ID Mapping
- `ID_MAPPING_MODE`: (`'file'`, `'regex'`, `'api'`, `'none'`)
    - `'file'`: **(Recommended for large datasets)** Uses a memory-efficient method to parse the large `idmapping_selected.tab` file and create a fast, on-disk lookup database.
    - `'regex'`: A faster but less comprehensive method that extracts UniProt IDs directly from FASTA headers using regular expressions.
    - `'api'`: Queries the UniProt API to map IDs. Very slow and subject to rate limits.
    - `'none'`: Performs no ID mapping.
- `API_MAPPING_FROM_DB`: The database to map from (e.g., `'UniRef50'`) when using `'api'` mode or creating the database in `'file'` mode.

#### Model Architecture
- `PROTGRAM_MODELS_TO_TRAIN`: (`list[str]`) A list of GCN variants to train on the ProtGram graphs. Options: `'directgcn'`, `'rgcn'`, `'tongdigcn'`.
- `GCN_HIDDEN_LAYER_DIMS`: (`list[int]`) A list defining the number of units in each hidden layer of the GCN.
- `GCN_1GRAM_INIT_DIM`: (`int`) The initial feature dimension for 1-gram nodes (amino acids).
- `GCN_USE_VECTOR_COEFFS`: (`True`/`False`) A key innovation of DirectGCN; if `True`, uses learned vector coefficients for message passing instead of scalar values.

#### Training Hyperparameters
- `GCN_EPOCHS_PER_LEVEL`: (`int`) Number of training epochs for each n-gram level.
- `GCN_LR`: (`float`) The learning rate for the GCN optimizer.
- `GCN_USE_LR_SCHEDULER`: (`True`/`False`) Enables a learning rate scheduler that reduces the LR on a plateau.
- `GCN_USE_EARLY_STOPPING`: (`True`/`False`) Enables early stopping to prevent overfitting.

#### Self-Supervised Tasks
- `GCN_TASK_TYPES_PER_LEVEL`: (`dict`) Maps each n-gram level (`int`) to a self-supervised task (`str`).
    - `'community'`: A task where the model predicts the community membership of nodes.
    - `'next_node'`: A task where the model predicts the next node in a random walk.

#### Cluster-GCN Strategy
- `GCN_USE_CLUSTER_TRAINING`: (`True`/`False`) If `True`, uses the Cluster-GCN strategy for very large graphs to make training feasible on a single GPU.
- `GCN_TARGET_NODES_PER_CLUSTER`: (`int`) The desired number of nodes in each graph cluster/subgraph.

#### Post-Processing
- `APPLY_PCA_TO_GCN`: (`True`/`False`) If `True`, applies Principal Component Analysis (PCA) to reduce the dimensionality of the final embeddings.
- `PCA_TARGET_DIMENSION`: (`int`) The target dimension for PCA reduction.

### 4. Baseline Embedding Pipeline Parameters
These sections control the alternative embedding generation methods used for comparison.

- **Word2Vec (`_setup_word2vec_params`)**:
    - `W2V_VECTOR_SIZE`: (`int`) The dimensionality of the Word2Vec embeddings.
    - `W2V_WINDOW`: (`int`) The context window size.
    - `APPLY_PCA_TO_W2V`: (`True`/`False`) Whether to apply PCA to the final Word2Vec embeddings.
- **Transformers (`_setup_transformer_params`)**:
    - `TRANSFORMER_MODELS_TO_RUN`: (`list[dict]`) A list of models from Hugging Face to run (e.g., `Rostlab/prot_bert`).
    - `TRANSFORMER_BASE_BATCH_SIZE`: (`int`) The base batch size for inference, which is scaled by a model-specific multiplier.
    - `APPLY_PCA_TO_TRANSFORMER`: (`True`/`False`) Whether to apply PCA to the final Transformer embeddings.
- **LSTM (`_setup_lstm_params`)**:
    - `LSTM_EMBEDDING_DIM`: (`int`) The dimensionality of the initial token embeddings.
    - `LSTM_HIDDEN_DIM`: (`int`) The number of units in the LSTM hidden layers.
    - `LSTM_TRAIN_SEQ_LEN`: (`int`) The length of the subsequences used for the next-character prediction task.

### 5. PPI Evaluation Parameters
This section configures the final link prediction task.

- `EVAL_EDGE_EMBEDDING_METHOD`: (`'concatenate'`, `'hadamard'`, etc.) The method used to combine the embeddings of two proteins to create an edge embedding.
- `EVAL_N_FOLDS`: (`int`) The number of folds for cross-validation.
- `EVAL_MLP_DENSE1_UNITS`, `EVAL_MLP_DROPOUT1_RATE`, etc.: Parameters defining the architecture of the downstream Multi-Layer Perceptron (MLP) classifier.
- `SAMPLE_NEGATIVE_PAIRS`: (`int` or `None`) If set, randomly samples this many negative pairs to balance the dataset. Useful if you have far more negative than positive interactions.

### 6. MLflow and Benchmarking Parameters
- `USE_MLFLOW`: (`True`/`False`) Enables or disables experiment tracking with MLflow.
- `MLFLOW_TRACKING_URI`: The location to store MLflow runs. Defaults to `results/mlruns`.
- `BENCHMARK_NODE_CLASSIFICATION_DATASETS`: (`list[str]`) A list of standard datasets (`'Cora'`, `'CiteSeer'`, etc.) to use for the GNN benchmarking suite.

---

## Running the Project

Once the installation and data setup are complete, you can execute the main pipeline.
