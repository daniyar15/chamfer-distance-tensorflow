#!/bin/bash

# 0. Change to the directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

# 1. Get TensorFlow flags
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# 2. Compile CUDA Kernels
echo "Compiling CUDA kernels..."
nvcc -c -o chamfer_kernels.cu.o chamfer_kernels.cu \
    ${TF_CFLAGS[@]} \
    -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -w --expt-relaxed-constexpr

# 3. Compile and link the C++ code with the CUDA kernels
echo "Compiling and Linking C++ Ops..."
g++ -std=c++14 -shared -o chamfer_plugin.so chamfer_ops.cc chamfer_kernels.cu.o \
    ${TF_CFLAGS[@]} -fPIC \
    -L/usr/local/cuda/lib64 -L/usr/local/cuda-12.8/lib64 \
    -lcudart ${TF_LFLAGS[@]} -O2 -w

# 4. Check if successful
if [ -f "chamfer_plugin.so" ]; then
    echo "Success! chamfer_plugin.so created."
else
    echo "Compilation failed."
fi