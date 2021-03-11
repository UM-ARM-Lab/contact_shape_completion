import ros_numpy
from contact_shape_completion.kinect_listener import DepthCameraListener
from gpu_voxel_planning_msgs.srv import CompleteShape, CompleteShapeResponse, CompleteShapeRequest
from shape_completion_training.voxelgrid import conversions
import rospy
from sensor_msgs.msg import PointCloud2
from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.utils.tf_utils import add_batch_to_dict


class ContactShapeCompleter:
    def __init__(self):
        self.robot_view = DepthCameraListener()
        self.model = None
        self.complete_shape = rospy.Service("complete_shape", CompleteShape, self.complete_shape_srv)
        self.new_free_sub = rospy.Subscriber("swept_freespace_pointcloud", PointCloud2,
                                             self.new_swept_freespace_callback)
        self.pointcloud_repub = rospy.Publisher("swept_volume_republisher", PointCloud2, queue_size=10)
        self.last_visible_vg = None
        self.model_runner = None


    def load_network(self, trial):
        # global model_runner
        # global model_evaluator
        if trial is None:
            print("Not loading any inference model")
            return
        self.model_runner = ModelRunner(training=False, trial_path=trial)

    def get_visible_vg(self):
        self.last_visible_vg = self.robot_view.get_visible_element()

    def complete_shape_srv(self, req: CompleteShapeRequest):
        # print(req)
        return CompleteShapeResponse()

    def new_swept_freespace_callback(self, pt_msg: PointCloud2):
        self.pointcloud_repub.publish(pt_msg)
        print("Point message received")
        elem = self.last_visible_vg
        if elem is None:
            print("No visible vg to update")
            return
        #
        transformed_cloud = self.robot_view.transform_pts_to_target(pt_msg)
        xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(transformed_cloud)
        vg = conversions.pointcloud_to_voxelgrid(xyz_array, scale=self.robot_view.scale,
                                                 origin=self.robot_view.origin,
                                                 add_trailing_dim=True, add_leading_dim=False,
                                                 )
        #
        elem['known_occ'] = vg

        self.robot_view.VG_PUB.publish_elem_cautious(elem)

    def infer_completion(self):
        inference = self.model_runner.model(add_batch_to_dict(self.last_visible_vg))
        self.robot_view.VG_PUB.publish_inference(inference)

        return inference
