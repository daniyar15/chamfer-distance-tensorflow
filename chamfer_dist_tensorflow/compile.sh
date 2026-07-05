#!/bin/bash
set -euo pipefail

# Logging helper: writes a timestamped error message to stderr
log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

# Trap any non-zero exit code for logging before the script exits
trap 'log_error "Compilation failed at line $LINENO with exit code $?"' ERR

# 0. Change to the directory of this script
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

echo "[INFO] Getting TensorFlow compile and link flags..."

# 1. Get TensorFlow flags
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# 2. Compile CUDA Kernels
echo "[INFO] Compiling CUDA kernels..."
# TODO add more --gencode targets for other GPU architectures
nvcc -c -o chamfer_kernels.cu.o chamfer_kernels.cu \
    ${TF_CFLAGS[@]} \
    -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -w --expt-relaxed-constexpr \
    -gencode arch=compute_80,code=sm_80

# 3. Compile and link the C++ code with the CUDA kernels
echo "[INFO] Compiling and Linking C++ Ops..."
g++ -std=c++14 -shared -o chamfer_plugin.so chamfer_ops.cc chamfer_kernels.cu.o \
    ${TF_CFLAGS[@]} -fPIC \
    -L/usr/local/cuda/lib64 -L/usr/local/cuda-12.8/lib64 \
    -lcudart ${TF_LFLAGS[@]} -O2 -w

# 4. Verify the output was created
if [ -f "chamfer_plugin.so" ]; then
    echo "[INFO] Success! chamfer_plugin.so created."
else
    echo "[ERROR] chamfer_plugin.so was not created." >&2
    exit 1
fi