from sensor_msgs import point_cloud2

from shape_completion_training.voxelgrid.metrics import chamfer_distance_pointcloud, distance_matrix
import numpy as np
import tensorflow as tf


def pt_cloud_distance(pt_msg_1, pt_msg_2):
    pts_1 = point_cloud2.read_points(pt_msg_1, field_names=('x', 'y', 'z'))
    pts_2 = point_cloud2.read_points(pt_msg_2, field_names=('x', 'y', 'z'))

    pts_1 = tf.cast([p for p in pts_1], tf.float32)
    pts_2 = tf.cast([p for p in pts_2], tf.float32)

    d = distance_matrix(pts_1[::10], pts_2)
    chamf_dist = chamfer_distance_pointcloud(pts_1[::10], pts_2)

    d_ab = tf.reduce_mean(tf.reduce_min(d, axis=1))
    d_ba = tf.reduce_mean(tf.reduce_min(d, axis=0))
    dmax_ab = tf.reduce_max(tf.reduce_min(d, axis=1))
    dmax_ba = tf.reduce_max(tf.reduce_min(d, axis=0))
    return chamf_dist
    # return dmax_ab + dmax_ba


