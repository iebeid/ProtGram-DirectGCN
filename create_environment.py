import subprocess
import sys
import argparse


def print_header(title):
    """Prints a formatted header."""
    border = "=" * (len(title) + 4)
    print(f"\n{border}")
    print(f"=== {title} ===")
    print(f"{border}")


def print_success(message):
    """Prints a success message."""
    print(f"\n--- {message} ---")


def print_error_and_exit(message):
    """Prints an error message and exits the script."""
    print(f"\n*** ERROR: {message} ***", file=sys.stderr)
    print("*** ABORTING SCRIPT. Please review the error messages above. ***", file=sys.stderr)
    sys.exit(1)


def run_command(command, description=""):
    """Runs a command in the shell, showing a description and streaming its output."""
    if description:
        print(f"\n>--- Running: {description} ---")
    print(f">--- Command: {command} ---")
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        if process.returncode != 0:
            print(f"\n*** Command failed with exit code {process.returncode} ***", file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print(f"\n*** ERROR: Command not found. Is Conda/Mamba installed and in your PATH? ***", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\n*** An unexpected error occurred: {e} ***", file=sys.stderr)
        return False


def check_mamba():
    """Checks if mamba is installed and available."""
    try:
        subprocess.run("mamba --version", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Conda environment with GPU support for TensorFlow and PyTorch.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("env_name", type=str, help="The name for the new Conda environment.")
    args = parser.parse_args()
    env_name = args.env_name

    # --- Configuration: All packages installed via Conda ---
    python_version = "3.11"

    # **MODIFICATION**: Added the 'pyg' channel specifically for PyTorch Geometric.
    # The order is important: it tells conda where to look first.
    channels = ["pyg", "pytorch", "nvidia", "conda-forge"]

    # All packages in a single list for a one-shot, robust installation.
    conda_packages = [f"python={python_version}", # GPU Frameworks
        "pytorch", "torchvision", "torchaudio", "pytorch-cuda=12.1", "tensorflow", # PyTorch Geometric (will now be found in the 'pyg' channel)
        "pytorch-geometric", # Other project dependencies
        "tqdm", "dask", "h5py", "matplotlib", "pandas", "pyarrow", "pyqt", "requests", "scikit-learn", "seaborn", "mlflow", "biopython", "networkx", "gensim", "python-louvain", "transformers"]

    print_header(f"Unified GPU Environment Setup for '{env_name}'")

    solver = "mamba" if check_mamba() else "conda"
    print(f"--- Using {solver.capitalize()} for installation. ---")

    # --- Single Installation Step ---
    print("\n[Step 1/1] Creating environment and installing all packages with Conda...")

    channel_flags = " ".join([f"-c {c}" for c in channels])
    package_list = " ".join(f'"{pkg}"' for pkg in conda_packages)  # Quote packages to handle special characters
    create_command = f"{solver} create --name {env_name} -y {channel_flags} {package_list}"

    if not run_command(create_command, "Unified Conda Environment Creation"):
        print_error_and_exit(f"Failed to create the Conda environment '{env_name}'. Please check the logs.")

    print_header("Environment creation complete!")
    print_success("All packages were installed successfully via Conda. ✅")
    print("To activate your new environment, run:\n")
    print(f"> conda activate {env_name}\n")
    print("You can then run the unit tests to verify the GPU setup for both frameworks.")
