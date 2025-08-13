# Data Preparation and Cleaning Guide

This document outlines the recommended preprocessing steps for the protein sequence data (e.g., `uniprot_sprot.fasta`) before it is used by the ProtGram-DirectGCN pipeline.

> [!IMPORTANT]
> The main pipeline **does not** perform these cleaning steps automatically. It assumes that the input FASTA files provided in the `data/sequences` directory have already been curated. Following these steps is crucial for ensuring the quality of the generated graphs and embeddings, and for the reproducibility of the results.

---

## 1. Rationale for Data Cleaning

Raw protein sequence databases like UniProt contain a significant amount of redundancy. This includes:
-   **Duplicate Entries**: Identical sequences with different identifiers.
-   **Fragments**: Incomplete protein sequences.
-   **Highly Similar Sequences**: Sequences from closely related organisms or isoforms that are nearly identical.

Using raw, uncleaned data can introduce several problems:
-   **Bias in N-gram Statistics**: Highly redundant sequences can skew the co-occurrence frequencies of n-grams, leading to a graph that does not accurately represent the general "language" of proteins.
-   **Computational Inefficiency**: Processing millions of redundant sequences increases the time and memory required for graph construction and training.
-   **Data Leakage in Evaluation**: If fragments of a protein used in the training set also appear in the test set, it can lead to overly optimistic and misleading evaluation metrics.

## 2. Recommended Tool: CD-HIT

A standard and highly efficient tool for clustering and removing redundant biological sequences is **CD-HIT** (`cd-hit`). It can quickly cluster a large database and produce a non-redundant set of representative sequences.

You can install CD-HIT via Conda into your `(base)` or a separate environment:
```sh
conda install -c bioconda cd-hit
```

## 3. Recommended Cleaning Procedure

The following steps describe how to generate a non-redundant protein sequence file using `uniprot_sprot.fasta` as an example.

1.  **Download the Raw Data**: Obtain the original FASTA file from its source.
    ```sh
    wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
    gunzip uniprot_sprot.fasta.gz
    ```

2.  **Run CD-HIT**: Use `cd-hit` to cluster the sequences and remove redundancy. A sequence identity threshold of 90% (`-c 0.9`) is a common choice that balances removing redundancy while preserving some sequence diversity.

    ```sh
    cd-hit -i uniprot_sprot.fasta -o uniprot_sprot_cleaned.fasta -c 0.9 -n 5 -M 0 -T 0
    ```
    -   `-i`: Input FASTA file.
    -   `-o`: Output path for the cleaned, non-redundant FASTA file.
    -   `-c 0.9`: Cluster sequences that are at least 90% identical.
    -   `-n 5`: Word size for clustering (recommended for this identity level).
    -   `-M 0`: Use unlimited memory (adjust if necessary).
    -   `-T 0`: Use all available CPU threads.

3.  **Use the Cleaned File**: Place the resulting `uniprot_sprot_cleaned.fasta` file into the `data/sequences/` directory of this project. The pipeline will then use this curated file for all subsequent steps.