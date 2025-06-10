import subprocess
import sys
import os
import shutil
from tqdm import tqdm


def print_header(title):
    """Prints a formatted header."""
    border = "=" * (len(title) + 4)
    print(f"\n{border}")
    print(f"=== {title} ===")
    print(f"{border}")


def print_step(current, total, description):
    """Prints a formatted step message."""
    print(f"\n[Step {current}/{total}] {description}")


def print_success(message):
    """Prints a success message."""
    print(f"\n--- {message} ---")


def print_error_and_exit(message):
    """Prints an error message and exits the script."""
    print(f"\n*** ERROR: {message} ***", file=sys.stderr)
    print("*** ABORTING SCRIPT. Please review the error messages above. ***", file=sys.stderr)
    sys.exit(1)


def run_command(command, description):
    """Runs a command in the shell and prints its description."""
    print(f"\n>--- Running: {command} ---")
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


def get_conda_base_prefix():
    """Finds the base directory of the Conda installation."""
    try:
        # Use shell=True to ensure conda command is found in the shell environment
        result = subprocess.run('conda info --base', shell=True, check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except Exception:
        print_error_and_exit("Could not determine Conda base directory. Is Conda installed and in your PATH?")


def is_env_created(env_name, conda_base_path):
    """Checks if a conda environment already exists."""
    env_path = os.path.join(conda_base_path, 'envs', env_name)
    return os.path.exists(env_path)


def install_pip_packages(env_name, packages, conda_base_path):
    """Installs a list of pip packages into the specified conda environment."""
    # Construct the path to pip.exe dynamically and robustly
    if platform.system() == "Windows":
        pip_executable = os.path.join(conda_base_path, 'envs', env_name, 'Scripts', 'pip.exe')
    else:  # Linux or macOS
        pip_executable = os.path.join(conda_base_path, 'envs', env_name, 'bin', 'pip')

    # Ensure the path is enclosed in quotes to handle spaces
    pip_executable = f'"{pip_executable}"'

    with tqdm(total=len(packages), desc="Installing Pip Packages") as pbar:
        for pkg in packages:
            # The package string might contain extra flags like '-f ...', so split it carefully
            command = f"{pip_executable} install {pkg}"
            if not run_command(command, f"Installing {pkg}"):
                print(f"\n--- WARNING: Failed to install pip package: {pkg}. Continuing... ---", file=sys.stderr)
            pbar.update(1)


# Main execution block
if __name__ == "__main__":
    import platform

    # --- Configuration ---
    env_name = "protgram-directgcn-10"  # Using the same name to continue where we left off
    channels = ["pytorch", "nvidia", "conda-forge", "defaults"]
    conda_packages = ["python=3.9", "pytorch=1.12.1", "torchvision", "torchaudio", "cudatoolkit=11.3.1", "cudnn=8.2.1", "tensorflow=2.10.0", "tqdm", "dask", "dask-expr", "h5py", "matplotlib", "pandas", "pyarrow", "pyqt",
        "requests", "scikit-learn", "seaborn", "mlflow", "biopython", "networkx"]
    pip_packages = ["torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html", "torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+cu113.html",
        "torch-cluster -f https://data.pyg.org/whl/torch-1.12.1+cu113.html", "torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html", "torch-geometric", "gensim", "python-louvain", "transformers",
        "torch-geometric-signed-directed"]

    print_header("Starting ProtDiGCN Environment Creation Script")

    solver = "mamba" if check_mamba() else "conda"
    print(f"--- Using {solver.capitalize()} for installation. ---")

    conda_base_path = get_conda_base_prefix()

    # === [Step 1/2] Create Environment and Install All Conda Packages ===
    if not is_env_created(env_name, conda_base_path):
        print_step(1, 2, f"Creating environment and installing all conda packages using '{solver}'...")
        create_command = f"{solver} create --name {env_name} -y {' '.join(['-c ' + c for c in channels])} {' '.join(conda_packages)}"
        if not run_command(create_command, "Conda Environment Creation"):
            print_error_and_exit("Failed to create the conda environment.")
        print_success("Base environment with all conda packages created successfully!")
    else:
        print_step(1, 2, f"Environment '{env_name}' already exists. Skipping Conda package installation.")

    # === [Step 2/2] Install Pip Packages ===
    print_step(2, 2, "Installing PyTorch Geometric and other pip packages...")
    install_pip_packages(env_name, pip_packages, conda_base_path)
    print_success("Pip packages installed successfully!")

    print_header("Environment creation complete!")
    print("To activate your new environment, run:\n")
    print(f"> conda activate {env_name}\n")
