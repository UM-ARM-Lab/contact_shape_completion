import tensorflow as tf

def get_assumed_occ(pred_occ, chss):
    """
    Computes a matrix of "known_occupied", assuming each chs is satisfied by the most likely voxel from the predicted
    occupied
    Args:
        pred_occ: the predicted occupancy <1, 64, 64, 64, 1>
        chss: the chss <n, 64, 64, 64, 1>
    Returns:
    """
    if chss is None:
        return tf.zeros(pred_occ.shape)

    a = chss * pred_occ
    maxs = tf.reduce_max(a, axis=[1, 2, 3, 4], keepdims=True)
    return tf.reduce_max(tf.cast(maxs == a, tf.float32), axis=0, keepdims=True)