import tensorflow as tf
from chamfer_dist_tensorflow import compute_distances, compute_distances_l1

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
    pc1_exp = tf.expand_dims(pc1, axis=2)  # [B, N, 1, 3]
    pc2_exp = tf.expand_dims(pc2, axis=1)  # [B, 1, M, 3]
    
    dist_mat = tf.reduce_sum(tf.square(pc1_exp - pc2_exp), axis=-1)  # [B, N, M]

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

def calculate_chamfer_distances_l1(pc1, pc2):
    """
    Calculates the Chamfer Distance using L1 (Manhattan) distance.
    
    Args:
        pc1: Tensor of shape [batch_size, N, 3]
        pc2: Tensor of shape [batch_size, M, 3]
        
    Returns:
        dist: Tensor of shape [batch_size]
    """
    # Compute pairwise L1 distances
    # Expand dimensions for broadcasting
    pc1_exp = tf.expand_dims(pc1, axis=2)  # [B, N, 1, 3]
    pc2_exp = tf.expand_dims(pc2, axis=1)  # [B, 1, M, 3]
    
    # Compute L1 distance: sum of absolute differences across the last dimension
    dist_mat = tf.reduce_sum(tf.abs(pc1_exp - pc2_exp), axis=-1)  # [B, N, M]

    # Find nearest neighbors and compute mean distances as before
    idx_1 = tf.argmin(dist_mat, axis=-1, output_type=tf.int32)
    min_dist_1 = tf.gather(dist_mat, idx_1, axis=2, batch_dims=2)

    dist_mat_t = tf.transpose(dist_mat, perm=[0, 2, 1])
    idx_2 = tf.argmin(dist_mat_t, axis=-1, output_type=tf.int32)
    min_dist_2 = tf.gather(dist_mat_t, idx_2, axis=2, batch_dims=2)

    chamfer_dist_l1 = (tf.reduce_mean(min_dist_1, axis=1) + 
                       tf.reduce_mean(min_dist_2, axis=1))
    return chamfer_dist_l1

@tf.function
def test_graph_mode_l2(xyz1, xyz2):
    dist1, dist2, idx1, idx2 = compute_distances(xyz1, xyz2)
    loss = tf.reduce_mean(dist1, axis=1) + tf.reduce_mean(dist2, axis=1)
    return tf.reduce_mean(loss)

@tf.function
def test_graph_mode_l1(xyz1, xyz2):
    dist1, dist2, idx1, idx2 = compute_distances_l1(xyz1, xyz2)
    loss = tf.reduce_mean(dist1, axis=1) + tf.reduce_mean(dist2, axis=1)
    return tf.reduce_mean(loss)

def main():
    # input data
    xyz1 = tf.random.uniform((8, 256, 3), minval=-1, maxval=1, dtype=tf.float32)
    xyz2 = tf.random.uniform((8, 256, 3), minval=-1, maxval=1, dtype=tf.float32)
    # xyz1 = [[[0.0, 0.0, 0.0],
    #          [1.0, 0.0, 0.0],
    #          [0.0, 1.0, 0.0],
    #          [0.0, 0.0, 1.0],
    #          [1.0, 1.0, 1.0]]]
    # xyz2 = [[[0.0, 0.0, 1.0],
    #          [0.5, 0.5, 0.5]]]

    xyz1 = tf.convert_to_tensor(xyz1, dtype=tf.float32)
    xyz2 = tf.convert_to_tensor(xyz2, dtype=tf.float32)
    
    print("\nTesting L2 Chamfer Distance...")
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


    print("\nTesting L1 Chamfer Distance...")
    with tf.GradientTape() as tape3:
        tape3.watch(xyz1)
        dist1_l1, dist2_l1, idx1_l1, idx2_l1 = compute_distances_l1(xyz1, xyz2)
        print(f"dist1_l1 shape: {dist1_l1.shape}, dist2_l1 shape: {dist2_l1.shape}")
        loss3 = tf.reduce_mean(dist1_l1, axis=1) + tf.reduce_mean(dist2_l1, axis=1)
        loss3 = tf.reduce_mean(loss3)
    print("L1 Chamfer Distance loss: ", loss3.numpy())
    grad3 = tape3.gradient(loss3, xyz1)
    print(f"Gradient shape: {grad3.shape}")

    with tf.GradientTape() as tape4:
        tape4.watch(xyz1)
        chamfer_distance_l1 = calculate_chamfer_distances_l1(xyz1, xyz2)
        loss4 = tf.reduce_mean(chamfer_distance_l1)
    print("L1 Chamfer Distance loss (reference implementation): ", loss4.numpy())
    grad4 = tape4.gradient(loss4, xyz1)
    print(f"Gradient shape: {grad4.shape}")
    print("Loss difference: ", abs(loss3.numpy() - loss4.numpy()))
    print("Max difference in gradients: ", tf.reduce_max(tf.abs(grad3 - grad4)).numpy())
    print("Mean difference in gradients: ", tf.reduce_mean(tf.abs(grad3 - grad4)).numpy())

    print("\nTesting graph mode...")
    loss = test_graph_mode_l2(xyz1, xyz2)
    print("L2 graph mode loss: ", loss.numpy())

    loss_l1 = test_graph_mode_l1(xyz1, xyz2)
    print("L1 graph mode loss: ", loss_l1.numpy())

    

if __name__ == '__main__':
    main()
