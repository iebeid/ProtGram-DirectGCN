import os

def print_header(title):
    """Prints a formatted header to the console."""
    border = "=" * (len(title) + 4)
    print(f"\n{border}\n### {title} ###\n{border}\n")


def save_pandas_dataframe_to_csv(results_df, output_dir):
    """Saves the benchmark results dataframe to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, 'gnn_benchmark_summary.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nBenchmark summary saved to: {results_path}")