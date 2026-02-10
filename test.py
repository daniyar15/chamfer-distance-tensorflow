import tensorflow as tf
from chamfer_dist_tensorflow import compute_distances

def calculate_chamfer_distances(pc1, pc2):
    """
    Calculates the Chamfer Distance using tf.argmin and tf.gather 
    to ensure strict gradient flow through the nearest neighbor.
    This is a reference implementation.
    
    Args:
        pc1: Tensor of shape [batch_size, N, 3]
        pc2: Tensor of shape [batch_size, M, 3]
        
    Returns:
        dist: Tensor of shape [batch_size]
    """
    pc1_sq = tf.reduce_sum(tf.square(pc1), axis=-1, keepdims=True)
    # pc2_sq: [B, M] -> [B, 1, M] (transposed for broadcasting)
    pc2_sq = tf.reduce_sum(tf.square(pc2), axis=-1)
    pc2_sq = tf.expand_dims(pc2_sq, axis=1)
    
    # Compute the intersection term 2*x*y using matrix multiplication
    # [B, N, 3] @ [B, 3, M] -> [B, N, M]
    interaction = tf.matmul(pc1, pc2, transpose_b=True)
    
    # Combine terms: ||x||^2 + ||y||^2 - 2*x*y
    # This results in shape [B, N, M] directly, saving ~3x memory
    dist_mat = pc1_sq + pc2_sq - 2 * interaction
    
    # Numerical stability: floating point errors can make dist slightly negative
    # Force non-negative to avoid NaN if you later take sqrt
    dist_mat = tf.nn.relu(dist_mat)

    # 2. Compute min distance from pc1 to pc2 (Forward)
    # Find indices of the nearest neighbors in pc2 for each point in pc1
    # Shape: [batch_size, N]
    idx_1 = tf.argmin(dist_mat, axis=-1, output_type=tf.int32)
    
    # Gather the specific distance values using the indices
    # axis=2 tells gather to look along the M dimension (columns)
    min_dist_1 = tf.gather(dist_mat, idx_1, axis=2, batch_dims=2)

    # 3. Compute min distance from pc2 to pc1 (Backward)
    # We transpose the matrix to reuse the same gather logic efficiently
    # Transposed Shape: [batch_size, M, N]
    dist_mat_t = tf.transpose(dist_mat, perm=[0, 2, 1])
    
    # Find indices of the nearest neighbors in pc1 for each point in pc2
    # Shape: [batch_size, M]
    idx_2 = tf.argmin(dist_mat_t, axis=-1, output_type=tf.int32)
    
    # Gather the specific distance values
    min_dist_2 = tf.gather(dist_mat_t, idx_2, axis=2, batch_dims=2)
    
    # 4. Sum the means of both directions
    chamfer_dist = (tf.reduce_mean(min_dist_1, axis=1) + 
                    tf.reduce_mean(min_dist_2, axis=1))
    return chamfer_dist

def main():
    # input data
    xyz1 = tf.random.uniform((16, 8192, 3), minval=-1, maxval=1, dtype=tf.float32)
    xyz2 = tf.random.uniform((16, 8192, 3), minval=-1, maxval=1, dtype=tf.float32)
    # xyz1 = [[[0.0, 0.0, 0.0],
    #          [1.0, 0.0, 0.0],
    #          [0.0, 1.0, 0.0],
    #          [0.0, 0.0, 1.0],
    #          [1.0, 1.0, 1.0]]]
    # xyz2 = [[[0.0, 0.0, 1.0],
    #          [0.5, 0.5, 0.5]]]

    xyz1 = tf.convert_to_tensor(xyz1, dtype=tf.float32)
    xyz2 = tf.convert_to_tensor(xyz2, dtype=tf.float32)
    
    with tf.GradientTape() as tape1:
        tape1.watch(xyz1)
        dist1, dist2, idx1, idx2 = compute_distances(xyz1, xyz2)
        print(f"dist1 shape: {dist1.shape}, dist2 shape: {dist2.shape}")
        loss1 = tf.reduce_mean(dist1, axis=1) + tf.reduce_mean(dist2, axis=1)
        loss1 = tf.reduce_mean(loss1)
    print("Chamfer Distance loss: ", loss1.numpy())
    grad1 = tape1.gradient(loss1, xyz1)
    print(f"Gradient shape: {grad1.shape}")

    with tf.GradientTape() as tape2:
        tape2.watch(xyz1)
        chamfer_distance = calculate_chamfer_distances(xyz1, xyz2)
        loss2 = tf.reduce_mean(chamfer_distance)
    print("Chamfer Distance loss (reference implementation): ", loss2.numpy())
    grad2 = tape2.gradient(loss2, xyz1)
    print(f"Gradient shape: {grad2.shape}")
    print("Loss difference: ", abs(loss1.numpy() - loss2.numpy()))
    print("Max difference in gradients: ", tf.reduce_max(tf.abs(grad1 - grad2)).numpy())
    print("Mean difference in gradients: ", tf.reduce_mean(tf.abs(grad1 - grad2)).numpy())


if __name__ == '__main__':
    main()
