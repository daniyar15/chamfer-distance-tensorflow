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

# 0.1 Determine TensorFlow's build-time CUDA version and auto-detect a matching
#     CUDA installation. Critical: the custom op must link against the same CUDA
#     runtime that TensorFlow was built with, otherwise runtime crashes like
#     "munmap_chunk(): invalid pointer" will occur.
# Precedence:
#   1. $CUDA_HOME env var (explicit user override)
#   2. A CUDA installation matching TF's build-time CUDA version
#   3. Derive from nvcc location
#   4. Well-known fallback paths
echo "[INFO] Getting TensorFlow build info..."
TF_CUDA_VER=$(python -c "
import tensorflow as tf
try:
    print(tf.sysconfig.get_build_info().get('cuda_version', ''))
except Exception:
    print('')
" 2>/dev/null || true)

if [[ -n "${CUDA_HOME:-}" ]]; then
    # User override — trust it
    CUDA_HOME="${CUDA_HOME}"
elif [[ -n "$TF_CUDA_VER" ]]; then
    # Look for a CUDA installation matching TF's version
    TF_CUDA_MAJOR="${TF_CUDA_VER%%.*}"
    for candidate in "/usr/local/cuda-${TF_CUDA_VER}" "/usr/local/cuda-${TF_CUDA_MAJOR}"; do
        if [[ -d "$candidate/lib64" ]]; then
            CUDA_HOME="$candidate"
            echo "[INFO] Found CUDA $TF_CUDA_VER (matching TensorFlow) at: $CUDA_HOME"
            break
        fi
    done
fi

if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc &> /dev/null; then
    # nvcc is typically at $CUDA_HOME/bin/nvcc
    CUDA_HOME=$(dirname -- "$(dirname -- "$(command -v nvcc)")")
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    # Fallback: try well-known locations
    for candidate in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-11; do
        if [[ -d "$candidate/lib64" ]]; then
            CUDA_HOME="$candidate"
            break
        fi
    done
fi

if [[ -z "${CUDA_HOME:-}" ]] || [[ ! -d "$CUDA_HOME/lib64" ]]; then
    log_error "CUDA toolkit not found. Set CUDA_HOME or install CUDA."
    exit 1
fi

echo "[INFO] Using CUDA from: $CUDA_HOME"

echo "[INFO] Getting TensorFlow compile and link flags..."

# 1. Get TensorFlow flags
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# 2. Compile CUDA Kernels
echo "[INFO] Compiling CUDA kernels..."

# Determine CUDA architecture flags.
# We generate SASS (cubin) for common GPU generations and embed PTX
# (code=compute_80) so the driver can JIT-compile for future architectures.
# Adjust or comment out lines for architectures you do not need.
# Auto-detect GPU compute capability via nvidia-smi, with a fallback list.
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=compute_cap --format=csv,noheader &> /dev/null 2>&1; then
    DETECTED_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    echo "[INFO] Detected GPU compute capability: ${DETECTED_CC}"
    CUDA_GENCODE_FLAGS=(
        -gencode "arch=compute_${DETECTED_CC},code=sm_${DETECTED_CC}"
    )
else
    echo "[INFO] Could not detect GPU. Using common architectures for CUDA 12+"
    CUDA_GENCODE_FLAGS=(
        -gencode arch=compute_75,code=sm_75   # Turing: RTX 2080 Ti, T4, Quadro RTX
        -gencode arch=compute_80,code=sm_80   # Ampere: A100, A6000, RTX 3090
        -gencode arch=compute_86,code=sm_86   # Ampere: RTX 3060/3070/3080, A40
        -gencode arch=compute_89,code=sm_89   # Ada Lovelace: RTX 4090, L40
        -gencode arch=compute_90,code=sm_90   # Hopper: H100, H200
        -gencode arch=compute_80,code=compute_80  # PTX fallback
    )
fi

nvcc -c -o chamfer_kernels.cu.o chamfer_kernels.cu \
    ${TF_CFLAGS[@]} \
    -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -w --expt-relaxed-constexpr \
    "${CUDA_GENCODE_FLAGS[@]}"

# 3. Compile and link the C++ code with the CUDA kernels
echo "[INFO] Compiling and Linking C++ Ops..."
g++ -shared -o chamfer_plugin.so chamfer_ops.cc chamfer_kernels.cu.o \
    ${TF_CFLAGS[@]} -fPIC \
    -L"$CUDA_HOME/lib64" \
    -lcudart ${TF_LFLAGS[@]} -O2 -w

# 4. Verify the output was created
if [ -f "chamfer_plugin.so" ]; then
    echo "[INFO] Success! chamfer_plugin.so created."
else
    echo "[ERROR] chamfer_plugin.so was not created." >&2
    exit 1
fi