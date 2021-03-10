from contact_shape_completion.kinect_listener import DepthCameraListener
from gpu_voxel_planning_msgs.srv import CompleteShape, CompleteShapeResponse, CompleteShapeRequest

import rospy
from sensor_msgs.msg import PointCloud2


class ContactShapeCompleter:
    def __init__(self):
        self.robot_view = DepthCameraListener()
        self.model = None
        self.complete_shape = rospy.Service("complete_shape", CompleteShape, self.complete_shape_srv)
        self.new_free_sub = rospy.Subscriber("swept_freespace_pointcloud", PointCloud2,
                                             self.new_swept_freespace_callback)
        self.pointcloud_repub = rospy.Publisher("swept_volume_republisher", PointCloud2, queue_size=10)

    def complete_shape_srv(self, req: CompleteShapeRequest):
        # print(req)
        return CompleteShapeResponse()

    def new_swept_freespace_callback(self, pt_msg: PointCloud2):
        self.pointcloud_repub.publish(pt_msg)
        print("Point message received")
