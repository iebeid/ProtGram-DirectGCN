import cupy as cp
import numpy as np

print("--- CUDA Verification with CuPy ---")

try:
    # 1. Get information about the current GPU device
    device = cp.cuda.Device(0)
    device.use()
    print(f"Successfully selected GPU 0: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
except cp.cuda.runtime.CUDARuntimeError as e:
    print("Error: Could not find or initialize a CUDA-enabled GPU.")
    print(f"Details: {e}")
    exit()

# 2. Create two large random matrices directly on the GPU
# CuPy's syntax is intentionally very similar to NumPy's
print("\nCreating two large random matrices directly on the GPU...")
matrix_a_gpu = cp.random.randn(2000, 3000, dtype=cp.float32)
matrix_b_gpu = cp.random.randn(3000, 2500, dtype=cp.float32)
print(f"Matrix A shape: {matrix_a_gpu.shape} (on GPU)")
print(f"Matrix B shape: {matrix_b_gpu.shape} (on GPU)")

# 3. Perform matrix multiplication on the GPU
print("\nPerforming matrix multiplication on the GPU...")
result_gpu = cp.dot(matrix_a_gpu, matrix_b_gpu)

# 4. Synchronize the device to wait for the computation to finish
# This is a good practice to ensure the operation is complete before proceeding
cp.cuda.runtime.deviceSynchronize()
print("Matrix multiplication complete.")
print(f"Result matrix shape: {result_gpu.shape} (on GPU)")


# 5. Transfer a small subset of the result back to the CPU (as a NumPy array) to print it
result_subset_cpu = cp.asnumpy(result_gpu[:2,:2])

print(f"\nVerification successful! A small subset of the result tensor:\n{result_subset_cpu}")
print("\nThis confirms that CUDA is working correctly for direct computation via CuPy.")


import cupy as cp
from nvidia import cudnn
import numpy as np

print("--- cuDNN Verification with NVIDIA's cudnn-python Library ---")

# 1. Print the version of the cuDNN library we are using
try:
    print(f"Found cuDNN version: {cudnn.backend.get_version_string()}")
except cudnn.backend.CuDNNError as e:
    print(f"Error initializing cuDNN. Please ensure it is installed correctly.")
    print(f"Details: {e}")
    exit()


# 2. Set up data parameters for a sample convolution
# (batch_size, channels, height, width)
input_shape = (1, 3, 32, 32)
# (num_output_filters, input_channels, kernel_height, kernel_width)
filter_shape = (16, 3, 3, 3)

# 3. Create input and filter data on the GPU using CuPy
print("\nCreating input and filter tensors on the GPU with CuPy...")
# Using float32, the most common datatype for this work
x = cp.random.rand(*input_shape).astype(cp.float32)
w = cp.random.rand(*filter_shape).astype(cp.float32)
print(f"Input tensor 'x' created with shape: {x.shape}")
print(f"Filter tensor 'w' created with shape: {w.shape}")

# 4. Create a handle to the cuDNN library context
handle = cudnn.create_handle()
print("\nCreated cuDNN handle.")

# 5. Define the computation graph for a convolution
# This is a low-level step where we describe the operation to cuDNN
graph = cudnn.pygraph(
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT
)

# Define the input and filter tensors for the graph
X = graph.tensor(name="X", dim=x.shape, stride=x.strides, data_type=x.dtype)
W = graph.tensor(name="W", dim=w.shape, stride=w.strides, data_type=w.dtype)

# Define the convolution operation itself
Y = graph.conv_fprop(name="conv1", image=X, weight=W, padding=[1,1], stride=[1,1])
Y.set_output(True).set_data_type(Y.get_data_type())
print("Defined a convolution operation graph.")

# 6. Build and execute the graph
print("Building and executing the cuDNN graph...")
graph.build([cudnn.heuristic_mode.A])
workspace = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)
# Prepare a dictionary to hold the actual data
variant_pack = {X: x, W: w}

# Execute the convolution
graph.execute(variant_pack, workspace)
print("Execution successful.")

# 7. Get the result
y_result = variant_pack[Y]
print(f"Output tensor 'y' received with shape: {y_result.shape}")
print("\nVerification successful! A cuDNN-accelerated convolution was executed directly.")
