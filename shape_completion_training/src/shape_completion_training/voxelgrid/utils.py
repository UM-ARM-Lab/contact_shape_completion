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


def inflate_voxelgrid(vg, inflation=1, is_batched=True):
    if not is_batched:
        vg = tf.expand_dims(vg, axis=0)

    filter = tf.constant(np.ones((3, 3, 3, 1, 1)), dtype=tf.float32)
    convolved = tf.nn.convolution(
        vg, filter, strides=None, padding='SAME', data_format=None,
        dilations=None, name=None
    )
    result = tf.clip_by_value(convolved, 0.0, 1.0)
    if not is_batched:
        result = tf.squeeze(result, axis=0)
    return result


