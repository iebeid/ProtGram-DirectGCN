import os
import platform
import subprocess
import sys
import argparse

# --- Configuration ---
PYTHON_VERSION = "3.11"
# We define the PyTorch versions to ensure consistency.
PYTORCH_VERSION = "2.4.0"
TORCHVISION_VERSION = "0.19.0"
PYTORCH_CUDA_SUFFIX = "cu121"  # Using CUDA 12.1 wheels is the most stable choice for both frameworks


# --- End Configuration ---

def create_setup_script(commands):
    """Creates a platform-specific shell script from a list of commands."""
    is_windows = platform.system() == "Windows"
    script_extension = ".bat" if is_windows else ".sh"
    script_filename = f"setup_script{script_extension}"

    with open(script_filename, "w") as f:
        if not is_windows:
            f.write("#!/bin/bash\n")
            f.write("set -e\n")
        for command in commands:
            f.write(command + "\n")

    if not is_windows:
        os.chmod(script_filename, 0o755)
    return script_filename


def run_script(script_filename):
    """Executes the setup script."""
    is_windows = platform.system() == "Windows"
    print(f"--- Starting Environment Setup using '{script_filename}' ---")
    try:
        executor = ['cmd', '/c'] if is_windows else []
        process = subprocess.Popen(
            executor + ([script_filename] if is_windows else [f"./{script_filename}"]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        if process.stdout:
            for line in process.stdout:
                print(line, end='')
        process.wait()

        if process.returncode != 0:
            print(f"\n--- Script failed with exit code {process.returncode} ---")
            sys.exit(process.returncode)
        else:
            print("\n--- Environment setup completed successfully! ---")

    except FileNotFoundError:
        print(f"Error: Could not find '{script_filename}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(script_filename):
            os.remove(script_filename)
            print(f"--- Cleaned up temporary script file: {script_filename} ---")


def check_conda_installed():
    """Checks if conda is installed."""
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True, text=True, shell=False)
        print("--- Conda is installed. ---")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("--- ERROR: Conda is not installed or not in your system's PATH. ---")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install packages into the active Conda environment.")
    parser.parse_args()
    print("--- Starting package installation into the currently active Conda environment. ---")

    if not check_conda_installed():
        sys.exit(1)

    system = platform.system()
    compiler_commands = []
    if system == "Linux":
        compiler_commands.append("conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y")
        compiler_commands.append("rm -rf ~/.config/pycuda")
    elif system == "Darwin":
        compiler_commands.append("conda install -c conda-forge clang_osx-64 clangxx_osx-64 -y")
        compiler_commands.append("rm -rf ~/.config/pycuda")

    # --- MODIFIED COMMAND SEQUENCE ---
    command_sequence = [
        "conda clean --all -y",
        "conda update --all -y",

        # Step 1: Install all non-framework packages and compilers with Conda first.
        # This is fast and robust.
        *compiler_commands,
        "conda install -c conda-forge dask tqdm biopython matplotlib scipy scikit-learn mlflow transformers=4.41.2 gensim python-louvain seaborn pycuda networkx=3.2.1 -y",

        # Step 2: Install TensorFlow and PyTorch with PIP. This ensures they bring their own, compatible CUDA libraries.
        "echo '--- Installing TensorFlow with its own CUDA libraries via pip ---'",
        "pip install \"tensorflow[and-cuda]\"",  # Let pip choose the best compatible version

        "echo '--- Installing PyTorch with its own CUDA libraries via pip ---'",
        (f"pip install torch=={PYTORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio --index-url https://download.pytorch.org/whl/{PYTORCH_CUDA_SUFFIX}"),

        # Step 3: Install PyG dependencies, pointing to the now-installed PyTorch version.
        "echo '--- Installing PyG dependencies ---'",
        (f"pip install pyg_lib torch-scatter torch-sparse torch-geometric -f "
         f"https://data.pyg.org/whl/torch-{PYTORCH_VERSION}+{PYTORCH_CUDA_SUFFIX}.html"),

        # Step 4: Verification and Cleanup
        "echo '--- Verifying installations ---'",
        'python -c "import tensorflow as tf; print(f\'TensorFlow found {len(tf.config.list_physical_devices(\\\'GPU\\\'))} GPUs\')"',
        'python -c "import torch; print(f\'PyTorch CUDA available: {torch.cuda.is_available()}\')"',
        "conda clean --all -y",
        "pip cache purge"
    ]

    script_file = create_setup_script(command_sequence)
    run_script(script_file)