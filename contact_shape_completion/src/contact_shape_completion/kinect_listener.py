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

SCALE = 0.007
# origin = (2.446 - scale * 32, -0.384 - scale * 32, 0.86 - scale * 32)
x_bounds = (0, 64)
# x_bounds = (20, 43)
y_bounds = (0, 64)
z_bounds = (0, 64)

# target_frame = "victor_root"


class DepthCameraListener:
    def __init__(self, voxelgrid_forward_shift=0, scale=SCALE):
        self.scale = scale
        # origin = (2.446 - scale * 32, -0.384 - scale * 32, 0.86 - scale * 32)
        self.x_bounds = (0, 64)
        # x_bounds = (20, 43)
        self.y_bounds = (0, 64)
        self.z_bounds = (0, 64)
        self.origin = (0, 0, 0)

        self.target_frame = "victor_root"

        self.point_cloud_creator = PointcloudCreator([i for i in range(1, 3)],
                                                     topic_prefix="/kinect2_victor_head/qhd/")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.VG_PUB = VoxelgridPublisher(frame=self.target_frame, scale=self.scale, origin=self.origin)
        self.voxelgrid_forward_shift = voxelgrid_forward_shift

    def transform_pts_to_target(self, pt_msg, target_frame=None):
        if target_frame is None:
            target_frame = self.target_frame
        timeout = 1.0
        tries_limit = 3
        for i in range(tries_limit):
            try:
                # trans = self.tf_buffer.lookup_transform(target_frame, pt_msg.header.frame_id,
                #                                         pt_msg.header.stamp, rospy.Duration(timeout))
                trans = self.tf_buffer.lookup_transform(target_frame, pt_msg.header.frame_id,
                                                        rospy.Time.now(), rospy.Duration(timeout))
                break
            except tf2_ros.tf2.LookupException as ex:
                rospy.logwarn(ex)
                rospy.logwarn("NOT TRANSFORMING")
                # return pt_msg
            except tf2.ExtrapolationException as ex:
                rospy.logwarn(ex)
                rospy.logwarn("NOT TRANSFORMING")
                # return pt_msg
        else:
            raise RuntimeError(f"Unable to transform to target frame {target_frame}")

        cloud_out = do_transform_cloud(pt_msg, trans)
        return cloud_out

    def get_visible_element(self, save_file=None):
        while self.point_cloud_creator.img_msgs_to_process is None:
            print("Waiting for kinect image")
            rospy.sleep(0.5)
        while (pt_msg := self.point_cloud_creator.filter_pointcloud()) is None:
            rospy.sleep(0.5)

         #pt_msg = self.point_cloud_creator.filter_pointcloud()
        if save_file is not None:
            with save_file.open('wb') as f:
                pt_msg.serialize(f)

        return self.voxelize_visible_element(pt_msg)

    def voxelize_visible_element(self, pt_msg):

        cloud_out = self.transform_pts_to_target(pt_msg)

        elem = self.voxelize_point_cloud(cloud_out)
        self.VG_PUB.origin = self.origin
        self.VG_PUB.publish_elem_cautious(elem)
        return elem

    def voxelize_point_cloud(self, pts):
        # global origin
        xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pts)
        occluding_xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pts)

        if len(xyz_array) != 0:
            self.origin = np.mean(xyz_array, axis=0) - np.array(
                [self.scale * 32 - self.voxelgrid_forward_shift, self.scale * 32, self.scale * 32])
        vg = conversions.pointcloud_to_voxelgrid(xyz_array, scale=self.scale, origin=self.origin,
                                                 add_trailing_dim=True, add_leading_dim=False)

        occluding_vg = conversions.pointcloud_to_voxelgrid(occluding_xyz_array, scale=self.scale, origin=self.origin,
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
        return elem
