import numpy as np


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


def voxelgrid_to_pointcloud(voxelgrid, scale=1.0, origin=[0, 0, 0], threshold=0.5):
    """
    Converts a 3D voxelgrid into a 3D set of points for each voxel with value above threshold
    @param voxelgrid: (opt 1 x) X x Y x Z (opt x 1) voxelgrid
    @param scale:
    @param origin: origin in voxel coorindates
    @param threshold:
    @return:
    """
    pts = np.argwhere(np.squeeze(voxelgrid) > threshold)
    return (np.array(pts) - origin + 0.5) * scale


def pointcloud_to_voxelgrid(pointcloud, scale=1.0, origin=[0, 0, 0], shape=(64, 64, 64),
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
    vg = np.zeros(shape)
    s = (pointcloud / scale + origin).astype(int)
    vg[s[:, 0], s[:, 1], s[:, 2]] = 1.0
    return add_extra_dims(vg, add_leading_dim, add_trailing_dim)


def transform_voxelgrid(vg, transform, scale=1.0, origin=(0, 0, 0)):
    """

    @param vg: voxelgrid
    @param transform: 4x4 homogeneous tramsform matrix
    @return:
    """
    vg, add_leading_dim, add_trailing_dim = strip_extra_dims(vg)

    pt_cloud = voxelgrid_to_pointcloud(vg)

    if pt_cloud.shape[0] == 0:
        return vg

    R = transform[0:3, 0:3]
    t = transform[0:3,3]
    trans_cloud = np.dot(R,pt_cloud.transpose()).transpose() + t
    return pointcloud_to_voxelgrid(trans_cloud, scale=scale, origin=origin, shape=vg.shape,
                                   add_trailing_dim=add_trailing_dim,
                                   add_leading_dim=add_leading_dim)

