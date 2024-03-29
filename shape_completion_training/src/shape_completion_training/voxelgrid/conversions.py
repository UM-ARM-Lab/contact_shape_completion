import numpy as np
from numpy import sin, cos
import tensorflow as tf


def strip_extra_dims(vg):
    """
    strips the leading and trailing dimensions from the voxelgrid
    @param vg:
    @return: stripped voxelgrid, has_leading_dim, has_ending_dim
    """
    return np.squeeze(vg), vg.shape[0] == 1, vg.shape[-1] == 1


def add_extra_dims(vg, add_leading_dim, add_trailing_dim):
    if add_leading_dim:
        vg = np.expand_dims(vg, 0)
    if add_trailing_dim:
        vg = np.expand_dims(vg, -1)
    return vg


def get_format(vg):
    return vg.shape[0] == 1, vg.shape[-1] == 1


def format_voxelgrid(voxelgrid, leading_dim, trailing_dim):
    squeeze = lambda x: np.squeeze(x)
    expand_dims = lambda x, axis: np.expand_dims(x, axis)
    if tf.is_tensor(voxelgrid):
        squeeze = lambda x: tf.squeeze(x)
        expand_dims = lambda x, axis: tf.expand_dims(x, axis)

    voxelgrid = squeeze(voxelgrid)
    if leading_dim:
        voxelgrid = expand_dims(voxelgrid, 0)
    if trailing_dim:
        voxelgrid = expand_dims(voxelgrid, -1)
    return voxelgrid


def pointcloud_to_sparse_voxelgrid(pointcloud, scale=1.0, origin=(0, 0, 0), shape=(64,64,64)):
    s = ((pointcloud - origin) / scale).astype(int)
    valid = np.logical_and(np.min(s, axis=1) >= 0, np.max(s, axis=1) < shape[0])
    sparse_vg = dict()
    for voxel in s[valid]:
        voxel = tuple(coord for coord in voxel)
        if voxel not in sparse_vg:
            sparse_vg[voxel] = 0
        sparse_vg[voxel] += 1
    return sparse_vg


def sparse_voxelgrid_to_pointcloud(sparse_vg: dict, scale=1.0, origin=(0,0,0), threshold=1):
    return np.array([(np.array(pt) + 0.5)*scale + origin for pt, count in sparse_vg.items() if count >= threshold])

def combine_sparse_voxelgrids(sparse_vg_1, sparse_vg_2):
    """
    NOTE: ASSUMES SAME ORIGIN AND SCALING
    Args:
        sparse_vg_1:
        sparse_vg_2:

    Returns:
    """
    for point, count in sparse_vg_2.items():
        if point not in sparse_vg_1:
            sparse_vg_1[point] = 0
        sparse_vg_1[point] += count
    return sparse_vg_1



def voxelgrid_to_pointcloud(voxelgrid, scale=1.0, origin=(0, 0, 0), threshold=0.5):
    """
    Converts a 3D voxelgrid into a 3D set of points for each voxel with value above threshold
    @param voxelgrid: (opt 1 x) X x Y x Z (opt x 1) voxelgrid
    @param scale:
    @param origin: origin in voxel coorindates
    @param threshold:
    @return:
    """
    pts = np.argwhere(np.squeeze(voxelgrid) > threshold)
    return (np.array(pts) + 0.5) * scale + origin
    # pts = tf.cast(tf.where(tf.squeeze(voxelgrid) > threshold), tf.float32)
    # return (pts - origin + 0.5) * scale


def pointcloud_to_voxelgrid(pointcloud, scale=1.0, origin=(0, 0, 0), shape=(64, 64, 64),
                            add_leading_dim=False, add_trailing_dim=False):
    """
    Converts a set of 3D points into a binary voxel grid
    @param pointcloud:
    @param scale: scale of the voxelgrid
    @param origin:
    @param shape:
    @param add_trailing_dim:
    @param add_leading_dim:
    @return:
    """
    vg = np.zeros(shape, np.float32)
    if tf.is_tensor(pointcloud):
        pointcloud = pointcloud.numpy()
    # I have used this before, but it is not symetric with voxelgrid_to_pointcloud
    s = ((pointcloud - origin) / scale).astype(int)
    # s = (pointcloud / scale + origin).astype(int)
    valid = np.logical_and(np.min(s, axis=1) >= 0, np.max(s, axis=1) < shape[0])
    s = s[valid]
    vg[s[:, 0], s[:, 1], s[:, 2]] = 1.0
    return format_voxelgrid(vg, add_leading_dim, add_trailing_dim)


def pointcloud_to_known_freespace_voxelgrid(pointcloud, scale=1.0, origin=(0, 0, 0), shape=(64, 64, 64),
                                            add_leading_dim=False, add_trailing_dim=False):
    """
    Computes the binary known freespace voxelgrid from a set of points
    NOTE! UNTESTED
    """
    vg = np.zeros(shape)
    if tf.is_tensor(pointcloud):
        pointcloud = pointcloud.numpy()
    # s = (pointcloud / scale + origin).astype(int)
    s = ((pointcloud - origin) / scale).astype(int)
    valid = np.logical_and(np.min(s, axis=1) >= 0, np.max(s, axis=1) < shape[0])
    s = s[valid]
    vg[s[:, 0], s[:, 1], s[:, 2]] = 1.0
    return format_voxelgrid(vg, add_leading_dim, add_trailing_dim)


def transform_voxelgrid(vg, transform, scale=1.0, origin=(0, 0, 0)):
    """
    @param vg: voxelgrid
    @param transform: 4x4 homogeneous tramsform matrix
    @return:
    """
    vg, add_leading_dim, add_trailing_dim = strip_extra_dims(vg)

    pt_cloud = voxelgrid_to_pointcloud(vg, scale=scale, origin=origin)

    if pt_cloud.shape[0] == 0:
        return vg

    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    trans_cloud = np.dot(R, pt_cloud.transpose()).transpose() + t
    return pointcloud_to_voxelgrid(trans_cloud, scale=scale, origin=origin, shape=vg.shape,
                                   add_trailing_dim=add_trailing_dim,
                                   add_leading_dim=add_leading_dim)


def make_transform(thetas=(0, 0, 0), translation=(0, 0, 0)):
    rot_x = [[1, 0, 0],
             [0, cos(thetas[0]), -sin(thetas[0])],
             [0, sin(thetas[0]), cos(thetas[0])]]
    rot_y = [[cos(thetas[1]), 0, sin(thetas[1])],
             [0, 1, 0],
             [-sin(thetas[1]), 0, cos(thetas[1])]]
    rot_z = [[cos(thetas[2]), -sin(thetas[2]), 0],
             [sin(thetas[2]), cos(thetas[2]), 0],
             [0, 0, 1]]
    R = np.dot(rot_x, np.dot(rot_y, rot_z))
    T = np.block([[R, np.expand_dims(np.array(translation), 0).transpose()], [np.zeros(3), 1]])
    return T


def downsample(voxelgrid, kernel_size=2):
    if kernel_size == 1:
        return voxelgrid

    if not tf.is_tensor(voxelgrid):
        voxelgrid = tf.cast(voxelgrid, tf.float32)
    leading, trailing = get_format(voxelgrid)
    formatted = format_voxelgrid(
        tf.nn.max_pool(format_voxelgrid(voxelgrid, True, True), ksize=kernel_size, strides=kernel_size,
                       padding="VALID"),
        leading, trailing)
    return formatted


def to_2_5D(voxelgrid, width=64):
    ind = tf.where(voxelgrid)
    img = tf.scatter_nd(ind[:,1:], ind[:,0], (width, width, 1))
    img = tf.cast(img, tf.float32)
    img = img + 64 * tf.cast(img == 0, tf.float32)
    return img


def img_to_voxelgrid(img, max_depth=64):
    ind = tf.where(tf.logical_and(0 <= img, img < max_depth))
    depths = tf.cast(tf.gather_nd(img, ind), tf.int64)
    depths = tf.expand_dims(depths, 1)

    ko_ind = tf.concat([depths, ind], 1)
    updates = tf.cast(ko_ind[:,0] * 0 + 1, tf.float32)
    return tf.scatter_nd(ko_ind, updates, (max_depth,max_depth,max_depth,1))
