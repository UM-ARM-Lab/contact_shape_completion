import numpy as np
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


def inflate_voxelgrid(vg, inflation=1):
    filter = tf.constant(np.ones((3, 3, 3, 1, 1)), dtype=tf.float32)
    convolved = tf.nn.convolution(
        vg, filter, strides=None, padding='SAME', data_format=None,
        dilations=None, name=None
    )
    return tf.clip_by_value(convolved, 0.0, 1.0)


