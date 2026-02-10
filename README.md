# TensorFlow Chamfer Distance (CUDA Optimized)

This package provides a robust implementation of the Chamfer Distance function for TensorFlow.
## The Problem

Standard TensorFlow implementations of Chamfer Distance often fail when the distance matrix between two point clouds exceeds `2,147,483,647` elements (the `2^31−1` limit for `int32`). In these cases, TensorFlow typically throws an `InvalidArgumentError` or an Out of Memory (`OOM`) error.

For instance, processing two batches of point clouds with dimensions of `32*8192*3` will trigger these failures. This is primarily due to `int32` indexing within internal ops; however, even forcing `int64` indexing often fails to resolve the underlying memory overhead.

## The Solution

This package bypasses these limitations by using a custom CUDA kernel to compute minimum distances and their corresponding indices.
### Installation & Usage

1) Compile the kernels: Run the provided compilation script:
```
./chamfer_dist_tensorflow/compile.sh
```

2) Testing: Refer to test.py for example usage and verification. Running the script will:

    - Compute the Chamfer distance between two batches of random point clouds.
    - Calculate gradients.
    - Validate the results against a reference implementation that uses standard TensorFlow ops.