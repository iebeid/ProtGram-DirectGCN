import os
import platform
import argparse


# This script generates a shell script to perform the installation.
# It does not install anything itself.

def main():
    parser = argparse.ArgumentParser(description="Generate a robust installation script for the ProtDiGCN environment.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("env_name", type=str, help="The name for the new Conda environment.")
    args = parser.parse_args()
    env_name = args.env_name

    # --- Configuration ---
    python_version = "3.11"
    torch_version = "2.3.1"
    cuda_version_for_pytorch = "cu121"

    # --- Define Packages for each step ---
    # Minimal conda packages
    base_conda_packages = [f"python={python_version}", "pip", "gxx_linux-64"]

    # Pip packages for PyTorch
    torch_pip_packages = f"torch=={torch_version} torchvision torchaudio --index-url https://download.pytorch.org/whl/{cuda_version_for_pytorch}"

    # Pip packages for PyG dependencies (with the crucial find-links flag)
    pyg_pip_packages = f"--find-links https://data.pyg.org/whl/torch-{torch_version}+{cuda_version_for_pytorch}.html torch-scatter torch-sparse torch-cluster torch-spline-conv"

    # All other remaining packages
    other_pip_packages = ["pytorch-geometric", "tensorflow[and-cuda]", "tqdm", "dask", "h5py", "matplotlib", "pandas", "pyarrow", "pyqt", "requests", "scikit-learn", "seaborn", "mlflow", "biopython", "networkx",
        "gensim", "python-louvain", "transformers", "torch-geometric-signed-directed"]

    # --- Generate the Shell Script ---
    # Using a .sh extension as the logs show a Linux environment
    script_filename = f"install_{env_name}.sh"

    with open(script_filename, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# This script was auto-generated to create the Conda environment.\n")
        f.write("# To run it, use the command: bash " + script_filename + "\n\n")

        f.write("# Stop on any error\n")
        f.write("set -e\n\n")

        f.write("# Make sure conda is initialized\n")
        f.write("eval \"$(conda shell.bash hook)\"\n\n")

        f.write("echo '--- [Step 1/4] Creating minimal Conda environment... ---\n'")
        f.write(f"conda create --name {env_name} -y -c conda-forge {' '.join(base_conda_packages)}\n\n")

        f.write("echo '--- [Step 2/4] Activating environment and installing PyTorch... ---\n'")
        f.write(f"conda activate {env_name}\n")
        f.write(f"pip install {torch_pip_packages}\n\n")

        f.write("echo '--- [Step 3/4] Installing PyTorch Geometric dependencies... ---\n'")
        f.write(f"pip install {pyg_pip_packages}\n\n")

        f.write("echo '--- [Step 4/4] Installing TensorFlow and all remaining packages... ---\n'")
        f.write(f"pip install {' '.join(other_pip_packages)}\n\n")

        f.write("echo '✅✅✅ Environment creation complete! ✅✅✅'\n")
        f.write("echo 'To activate your new environment, run:'\n")
        f.write(f"echo 'conda activate {env_name}'\n")

    # Make the generated script executable
    os.chmod(script_filename, 0o755)

    print("=" * 60)
    print("✅ Generated new installer script!")
    print(f"   To create your environment, please run the following command in your terminal:")
    print(f"\n   bash ./{script_filename}\n")
    print("=" * 60)


if __name__ == "__main__":
    main()
