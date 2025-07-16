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

## Data Setup

The project is configured to automatically download required data files, such as the UniProt FASTA database.

1.  **Review Data Sources**: Before running the main pipeline, open `configuration/config.py`. Inside the `DATA_SOURCES` dictionary, you will find URLs for various data files.
2.  **Update Placeholder URLs**: Some URLs for interaction data (`POS_INTERACTIONS`, `NEG_INTERACTIONS`) are placeholders. **You must replace `"https://example.com/..."` with the actual URLs for your data.**
3.  **Run the Pipeline**: When you run the main project pipeline, the `DataManager` will automatically check for the required files, download them from the URLs you provided, and decompress them if necessary.

## Running the Project

Once the installation and data setup are complete, you can execute the main pipeline.

```sh
# (Assuming you are in the project's root directory)
python main.py
```