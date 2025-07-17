import os
import platform
import subprocess
import sys
import argparse

# --- Configuration ---
PYTHON_VERSION = "3.11"
CUDA_TOOLKIT_VERSION = "12.5"
CUDNN_VERSION = "9.3"

# Define PyTorch versions to align with the CUDA toolkit
# For CUDA 12.5 from conda, PyTorch uses the cu124 wheels.
PYTORCH_VERSION = "2.4.0" # A recent version compatible with CUDA 12.x
TORCHVISION_VERSION = "0.19.0"
PYTORCH_CUDA_SUFFIX = "cu124"


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
            executor + ([script_filename] if is_windows else [f"./{script_filename}"]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Stream the output in real-time
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
        print(f"Error: Could not find '{script_filename}'. Please ensure it was created correctly.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)
    # Clean up the generated script file
    if os.path.exists(script_filename):
        os.remove(script_filename)
        print(f"--- Cleaned up temporary script file: {script_filename} ---")


def check_conda_installed():
    """Checks if conda is installed and available in the system's PATH."""
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True, text=True, shell=False)
        print("--- Conda is installed. ---")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("--- ERROR: Conda is not installed or not in your system's PATH. ---")
        print("Please install Miniconda or Anaconda and try again.")
        print("Installation instructions: https://docs.conda.io/projects/miniconda/en/latest/")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Install required project packages into the currently active Conda environment. "
                    "Please ensure you have created and activated a suitable environment first (e.g., 'conda activate my-env')."
    )
    parser.parse_args()  # No arguments needed, but this allows for --help
    print("--- Starting package installation into the currently active Conda environment. ---")

    if not check_conda_installed():
        sys.exit(1)

    # Platform-specific step to ensure compilers are available in the environment.
    # This prevents build errors for packages that need to be compiled from source.
    system = platform.system()
    compiler_commands = []
    if system == "Linux":
        # For Linux, install the GNU compiler toolchain from conda-forge.
        print("--- Adding commands to install GCC/G++ compilers for Linux. ---")
        compiler_commands.append("conda install -c conda-forge gcc_linux-64 gxx_linux-64 -y")
        # Also clear any stale PyCUDA cache that might point to the wrong compiler.
        compiler_commands.append("echo '--- Clearing PyCUDA cache to prevent stale compiler paths ---'")
        compiler_commands.append("rm -rf ~/.config/pycuda")
    elif system == "Darwin":  # This is macOS
        # For macOS, install the Clang compiler toolchain from conda-forge.
        print("--- Adding commands to install Clang compilers for macOS. ---")
        compiler_commands.append("conda install -c conda-forge clang_osx-64 clangxx_osx-64 -y")
        # Also clear any stale PyCUDA cache that might point to the wrong compiler.
        compiler_commands.append("echo '--- Clearing PyCUDA cache to prevent stale compiler paths ---'")
        compiler_commands.append("rm -rf ~/.config/pycuda")
    elif system == "Windows":
        # For Windows, compilation often requires the MSVC build tools, which are
        # best installed manually via the Visual Studio Installer.
        print("\n--- INFO: On Windows, some packages may require the Microsoft C++ Build Tools. ---")
        print("--- If you encounter compilation errors, please install them and try again. ---\n")

    # This command sequence will install all packages into the active environment.
    command_sequence = [
        # Initial cleanup and update of the active environment
        "conda clean --all -y",
        "conda update --all -y",
        "conda clean --all -y",

        # Install core GPU libraries (CUDA, cuDNN)
        f"conda install -c nvidia cuda-toolkit={CUDA_TOOLKIT_VERSION} -y",
        f"conda install -c nvidia cudnn={CUDNN_VERSION} -y",

        # The platform-specific compiler commands will be inserted here

        # Install and verify TensorFlow
        "conda install -c conda-forge tensorflow -y",
        # Install tf-keras for backwards compatibility with Keras 2, required by transformers
        "echo '--- Installing tf-keras for Keras 2 API compatibility ---'",
        "pip install tf-keras",
        'python -c "import tensorflow as tf; print(\'Num GPUs Available: \', len(tf.config.list_physical_devices(\'GPU\')))"',

        # Install PyTorch using a specific index to match the conda-installed CUDA version. This is more robust.
        (f"pip install torch=={PYTORCH_VERSION} torchvision=={TORCHVISION_VERSION} torchaudio "
         f"--index-url https://download.pytorch.org/whl/{PYTORCH_CUDA_SUFFIX}"),
        'python -c "import torch; print(f\'PyTorch CUDA available: {torch.cuda.is_available()}\')"',

        # Final cleanup
        "conda clean --all -y",
        "pip cache purge",

        # Install remaining data science and ML libraries
        "conda install -c conda-forge dask -y",
        "conda install -c conda-forge tqdm -y",
        "conda install -c conda-forge biopython -y",
        # Install PyG dependencies pointing to the correct torch/cuda version
        (f"pip install pyg_lib torch-scatter torch-sparse -f "
         f"https://data.pyg.org/whl/torch-{PYTORCH_VERSION}+{PYTORCH_CUDA_SUFFIX}.html"),
        "conda install -c conda-forge matplotlib -y",
        "conda install -c conda-forge scipy -y",
        "conda install -c conda-forge scikit-learn -y",
        "pip install mlflow",
        "conda install -c conda-forge transformers -y",
        "conda install -c conda-forge gensim -y",
        "conda install -c conda-forge python-louvain -y",
        "pip install torch_geometric", # Now install the main package
        "conda install -c conda-forge seaborn -y",
        "conda install -c conda-forge pycuda -y",

        "conda clean --all -y",
        "pip cache purge"
    ]

    # Insert the compiler installation commands at the right place in the sequence.
    # This happens after core setup and before packages that might need compilation.
    command_sequence[5:5] = compiler_commands

    # Create the platform-specific script
    script_file = create_setup_script(command_sequence)

    # Execute the script
    run_script(script_file)