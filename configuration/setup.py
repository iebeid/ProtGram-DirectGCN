import os
import platform
import subprocess
import sys
import argparse

# --- Script Configuration ---
PYTHON_VERSION = "3.11"
CUDA_TOOLKIT_VERSION = "12.5"
CUDNN_VERSION = "9.3"


# --- End Configuration ---

def create_setup_script(commands):
    """Creates a platform-specific shell script from a list of commands."""
    is_windows = platform.system() == "Windows"
    script_extension = ".bat" if is_windows else ".sh"
    script_filename = f"setup_script{script_extension}"

    with open(script_filename, "w") as f:
        if not is_windows:
            # Add shebang for Linux/macOS
            f.write("#!/bin/bash\n")
            # Exit on any error
            f.write("set -e\n")

        # Add commands to the script
        for command in commands:
            f.write(command + "\n")

    # Make the script executable on non-Windows systems
    if not is_windows:
        os.chmod(script_filename, 0o755)

    return script_filename


def run_script(script_filename):
    """Executes the setup script."""
    is_windows = platform.system() == "Windows"

    print(f"--- Starting Environment Setup using '{script_filename}' ---")

    try:
        # For Windows, use 'cmd /c', for others, execute directly
        executor = ['cmd', '/c'] if is_windows else []

        process = subprocess.Popen(
            executor + [script_filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream the output in real-time
        for line in process.stdout:
            print(line, end='')

        process.wait()

        if process.returncode != 0:
            print(f"\n--- Script failed with exit code {process.returncode} ---")
            sys.exit(process.returncode)
        else:
            print("\n--- Environment setup completed successfully! ---")

    except FileNotFoundError:
        print(f"Error: Could not find '{script_filename}'. Please ensure it was created correctly.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        # Clean up the generated script file
        if os.path.exists(script_filename):
            os.remove(script_filename)
            print(f"--- Cleaned up temporary script file: {script_filename} ---")


if __name__ == "__main__":
    # Set up argument parser to accept the environment name
    parser = argparse.ArgumentParser(description="Create a Conda environment with specified packages.")
    parser.add_argument("env_name", type=str, help="The name for the new Conda environment.")
    args = parser.parse_args()

    # The environment name is now taken from the command-line argument
    env_name = args.env_name
    print(f"--- Target Environment Name: {env_name} ---")

    # The list of commands in the exact sequence you provided
    command_sequence = [
        # Initial cleanup and update
        "conda clean --all -y",
        "conda update --all -y",
        "conda clean --all -y",

        # Create and activate the new environment using the provided name
        f"conda create -n {env_name} python={PYTHON_VERSION} -y",
        f"conda activate {env_name}",

        # Install core GPU libraries (CUDA, cuDNN)
        f"conda install -c nvidia cuda-toolkit={CUDA_TOOLKIT_VERSION} -y",
        f"conda install -c nvidia cudnn={CUDNN_VERSION} -y",

        # Install and verify TensorFlow
        "conda install -c conda-forge tensorflow -y",
        'python -c "import tensorflow as tf; print(\'Num GPUs Available: \', len(tf.config.list_physical_devices(\'GPU\')))"',

        # Install and verify PyTorch
        "pip3 install torch torchvision torchaudio",
        'python -c "import torch; print(f\'PyTorch CUDA available: {torch.cuda.is_available()}\')"',

        # Final cleanup
        "conda clean --all -y",
        "pip cache purge",

        # Install remaining data science and ML libraries
        "conda install -c conda-forge dask -y",
        "conda install -c conda-forge tqdm -y",
        "conda install -c conda-forge biopython -y",
        "pip install torch_geometric",
        "conda install -c conda-forge matplotlib -y",
        "conda install -c conda-forge scipy -y",
        "conda install -c conda-forge scikit-learn -y",
        "pip install mlflow",
        "conda install -c conda-forge transformers -y",
        "conda install -c conda-forge gensim -y",
        "conda install -c conda-forge python-louvain -y",
        "pip install seaborn",
        "pip install pycuda"
    ]

    # Create the platform-specific script
    script_file = create_setup_script(command_sequence)

    # Execute the script
    run_script(f"./{script_file}" if platform.system() != "Windows" else script_file)