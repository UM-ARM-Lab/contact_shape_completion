import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2

from arc_utilities import ros_helpers
from rviz_voxelgrid_visuals import conversions as visual_conversions


def scene1_gt():
    vg = np.ones((5, 9, 17))

    pts = visual_conversions.vox_to_pointcloud2_msg(vg, scale=0.02, frame='gpu_voxel_world', origin=(-72, -100, -65),
                                                    density_factor=2)
    return pts


def visualize(pts, point_pub: rospy.Publisher):
    point_pub.publish(pts)


def main():
    rospy.init_node("simulation_scene")
    point_pub = ros_helpers.get_connected_publisher('/pointcloud', PointCloud2, queue_size=1)
    pts = scene1_gt()

    visualize(pts, point_pub)
    rospy.sleep(1)


if __name__ == "__main__":
    main()
