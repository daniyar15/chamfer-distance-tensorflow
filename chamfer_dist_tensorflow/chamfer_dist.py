import tensorflow as tf
import os

# Get the path to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load the library with the NEW name
plugin_path = os.path.join(BASE_DIR, "chamfer_plugin.so")
_chamfer_module = tf.load_op_library(plugin_path)


def compute_distances(
    xyz1: tf.Tensor, xyz2: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Args:
        xyz1: (B, N, 3)
        xyz2: (B, M, 3)

    Returns:
        A tuple containing
            - dist1: (B, N) l2 distances from xyz1 to xyz2
            - dist2: (B, M) l2 distances from xyz2 to xyz1
            - idx1: (B, N) nearest neighbor indices from xyz1 to xyz2
            - idx2: (B, M) nearest neighbor indices from xyz2 to xyz1
    """
    return _chamfer_module.chamfer_dist(xyz1, xyz2)


@tf.RegisterGradient("ChamferDist")
def _compute_distances_grad(op, grad_dist1, grad_dist2, grad_idx1, grad_idx2):
    xyz1 = op.inputs[0]
    xyz2 = op.inputs[1]
    idx1 = op.outputs[2]
    idx2 = op.outputs[3]

    if grad_dist1 is None:
        grad_dist1 = tf.zeros_like(op.outputs[0])
    if grad_dist2 is None:
        grad_dist2 = tf.zeros_like(op.outputs[1])

    grad_xyz1, grad_xyz2 = _chamfer_module.chamfer_dist_grad(
        xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2
    )

    return [grad_xyz1, grad_xyz2]
