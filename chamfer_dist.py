import tensorflow as tf
import os

# Get the path to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load the library with the NEW name
plugin_path = os.path.join(BASE_DIR, 'chamfer_plugin.so')
_chamfer_module = tf.load_op_library(plugin_path)

def chamfer_dist(xyz1, xyz2):
    """
    Args:
        xyz1: (B, N, 3)
        xyz2: (B, M, 3)
    """
    return _chamfer_module.chamfer_dist(xyz1, xyz2)

@tf.RegisterGradient("ChamferDist")
def _chamfer_dist_grad(op, grad_dist1, grad_dist2, grad_idx1, grad_idx2):
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