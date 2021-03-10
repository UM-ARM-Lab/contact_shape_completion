import tensorflow as tf


def sample_from_voxelgrid(vg, n=1):
    """
    Samples a voxelgrid with n  occupied voxels from the occupied voxels of vg

    Args:
        vg:

    Returns:

    """

    # sample = tf.zeros(vg.shape)
    sampled_ind = tf.random.shuffle(tf.where(vg))[0]
    return tf.scatter_nd([sampled_ind], [1.0], vg.shape)
