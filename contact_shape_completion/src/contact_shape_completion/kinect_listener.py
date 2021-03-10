import ros_numpy
import rospy
import tf2_ros
import tf2_py as tf2
from sensor_msgs.msg import PointCloud2
import message_filters
from sensor_msgs.msg import CompressedImage
from object_segmentation.pointcloud_utils import PointcloudCreator
from shape_completion_visualization.voxelgrid_publisher import VoxelgridPublisher
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import numpy as np
from shape_completion_training.voxelgrid import conversions
from shape_completion_training.utils import data_tools



scale = 0.007
# origin = (2.446 - scale * 32, -0.384 - scale * 32, 0.86 - scale * 32)
x_bounds = (0, 64)
# x_bounds = (20, 43)
y_bounds = (0, 64)
z_bounds = (0, 64)

target_frame = "victor_root"


def voxelize_point_cloud(pts):
    # global origin
    xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pts)
    occluding_xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pts)

    if len(xyz_array) == 0:
        origin = (0, 0, 0)
    else:
        origin = np.mean(xyz_array, axis=0) - np.array([scale * 32 - .04, scale * 32, scale * 32])
    vg = conversions.pointcloud_to_voxelgrid(xyz_array, scale=scale, origin=origin,
                                             add_trailing_dim=True, add_leading_dim=False)

    occluding_vg = conversions.pointcloud_to_voxelgrid(occluding_xyz_array, scale=scale, origin=origin,
                                                       add_trailing_dim=True, add_leading_dim=False)

    # vg = crop_vg(vg)
    # vg = np.swapaxes(vg, 1,2)
    elem = {'known_occ': vg}
    ko, kf = data_tools.simulate_2_5D_input(vg)
    _, kf = data_tools.simulate_2_5D_input(occluding_vg)

    # if "YCB" in trial:
    # if True:
        # ko, kf = data_tools.simulate_slit_occlusion(ko, kf, x_bounds[0], x_bounds[1])
        # nz = np.nonzero(ko)
        # try:
        #     lb = min(nz[1])
        #     ub = max(nz[1]) - 1
        # except ValueError:
        #     lb = 0
        #     ub = 63
        # kf[:, 0:lb, :, 0] = 0
        # kf[:, ub:, :, 0] = 0

    elem['known_occ'] = ko
    elem['known_free'] = kf
    return elem, origin


class DepthCameraListener:
    def __init__(self):
        self.point_cloud_creator = PointcloudCreator([i for i in range(1, 25)],
                                                     topic_prefix="/kinect2_victor_head/qhd/")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.VG_PUB = VoxelgridPublisher(frame=target_frame, scale=scale, origin=(0, 0, 0))




    def get_visible_element(self):
        while self.point_cloud_creator.img_msgs_to_process is None:
            print("Waiting for kinect image")
            rospy.sleep(0.5)
        while self.point_cloud_creator.filter_pointcloud() is None:
            rospy.sleep(0.5)

        pt_msg = self.point_cloud_creator.filter_pointcloud()

        timeout = 1.0
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, pt_msg.header.frame_id,
                                                    pt_msg.header.stamp, rospy.Duration(timeout))
        except tf2_ros.tf2.LookupException as ex:
            rospy.logwarn(ex)
            return
        except tf2.ExtrapolationException as ex:
            rospy.logwarn(ex)
            return

        cloud_out = do_transform_cloud(pt_msg, trans)
        elem, my_origin = voxelize_point_cloud(cloud_out)
        self.VG_PUB.origin = my_origin
        self.VG_PUB.publish_elem_cautious(elem)