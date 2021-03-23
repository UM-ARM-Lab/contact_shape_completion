import ros_numpy
from contact_shape_completion.kinect_listener import DepthCameraListener
from gpu_voxel_planning_msgs.srv import CompleteShape, CompleteShapeResponse, CompleteShapeRequest, RequestShape, \
    RequestShapeResponse, RequestShapeRequest
from shape_completion_training.voxelgrid import conversions
import rospy
from sensor_msgs.msg import PointCloud2
from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.utils.tf_utils import add_batch_to_dict
import tensorflow as tf

from rviz_voxelgrid_visuals import conversions as visual_conversions


class ContactShapeCompleter:
    def __init__(self):
        self.robot_view = DepthCameraListener()
        self.model = None
        self.complete_shape = rospy.Service("complete_shape", CompleteShape, self.complete_shape_srv)
        self.request_shape = rospy.Service("get_shape", RequestShape, self.request_shape_srv)
        self.new_free_sub = rospy.Subscriber("swept_freespace_pointcloud", PointCloud2,
                                             self.new_swept_freespace_callback)
        self.pointcloud_repub = rospy.Publisher("swept_volume_republisher", PointCloud2, queue_size=10)
        self.last_visible_vg = None
        self.model_runner = None
        self.swept_freespace = tf.zeros((1, 64, 64, 64, 1))

    def load_network(self, trial):
        # global model_runner
        # global model_evaluator
        if trial is None:
            print("Not loading any inference model")
            return
        self.model_runner = ModelRunner(training=False, trial_path=trial)

    def get_visible_vg(self):
        self.last_visible_vg = self.robot_view.get_visible_element()

    def request_shape_srv(self, req: RequestShapeRequest):
        if self.model_runner is None:
            raise AttributeError("Model must be loaded before inferring completion")

        inference = self.model_runner.model(add_batch_to_dict(self.last_visible_vg))
        pt = self.transform_to_gpuvoxels(self.last_visible_vg['known_occ'])
        # resp = RequestShapeResponse()
        # resp.points
        return RequestShapeResponse(points=pt)

    def complete_shape_srv(self, req: CompleteShapeRequest):
        # print(req)
        if self.model_runner is None:
            raise AttributeError("Model must be loaded before inferring completion")

        inference = self.model_runner.model(add_batch_to_dict(self.last_visible_vg))

        # pt = conversions.voxelgrid_to_pointcloud(inference['predicted_occ'])
        pt = self.transform_to_gpuvoxels(inference['predicted_occ'])
        resp = CompleteShapeResponse()
        resp.sampled_completions.append(pt)
        return resp

    def transform_from_gpuvoxels(self, pt_msg: PointCloud2):
        transformed_cloud = self.robot_view.transform_pts_to_target(pt_msg)
        xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(transformed_cloud)
        # TODO: visual_conversions produces the wrong result cause the transforms are calculated differently.
        #  Look into this
        vg = conversions.pointcloud_to_voxelgrid(xyz_array, scale=self.robot_view.scale,
                                                 origin=self.robot_view.origin,
                                                 add_trailing_dim=True, add_leading_dim=False,
                                                 )
        return vg

    def transform_to_gpuvoxels(self, vg) -> PointCloud2:
        pt_cloud = conversions.voxelgrid_to_pointcloud(vg, scale=self.robot_view.scale,
                                                       origin=self.robot_view.origin)

        # TODO: There is some diaster going on with the origin definition here. The problem is my tools
        #  visual_conversions and conversions define the origin differently
        msg = visual_conversions.vox_to_pointcloud2_msg(vg, frame=self.robot_view.target_frame,
                                                        scale=self.robot_view.scale,
                                                        origin=-self.robot_view.origin / self.robot_view.scale,
                                                        density_factor=3)
        # pt = conversions.voxelgrid_to_pointcloud(vg, scale=self.robot_view.scale,
        #                                          origin=self.robot_view.origin)

        msg = self.robot_view.transform_pts_to_target(msg, target_frame="gpu_voxel_world")
        return msg

    def new_swept_freespace_callback(self, pt_msg: PointCloud2):
        self.pointcloud_repub.publish(pt_msg)
        print("Point message received")
        elem = self.last_visible_vg
        if elem is None:
            print("No visible vg to update")
            return
        #
        vg = self.transform_from_gpuvoxels(pt_msg)
        # visual_vg = visuals_conversions.pointcloud2_msg_to_vox(transformed_cloud, scale=self.robot_view.scale, origin=self.robot_view.origin)
        #
        # elem['known_occ'] = vg
        self.swept_freespace = vg
        self.robot_view.VG_PUB.publish_elem_cautious(elem)

    def infer_completion(self):
        if self.model_runner is None:
            raise AttributeError("Model must be loaded before inferring completion")

        inference = self.model_runner.model(add_batch_to_dict(self.last_visible_vg))

        # inference['predicted_occ'] -= self.swept_freespace

        latent = tf.Variable(self.model_runner.model.sample_latent(add_batch_to_dict(self.last_visible_vg)))
        predicted_occ = self.model_runner.model.decode(latent, apply_sigmoid=True)
        self.robot_view.VG_PUB.publish('predicted_occ', predicted_occ)

        self.enforce_contact(latent)

        # TODO Graident decent to remove swept freespace

        return inference

    def enforce_contact(self, latent):
        pssnet = self.model_runner.model
        self.robot_view.VG_PUB.publish('predicted_occ', pssnet.decode(latent, apply_sigmoid=True))
        known_contact = tf.Variable(tf.zeros((1, 64, 64, 64, 1)))
        # known_free = tf.Variable(tf.zeros((1, 64, 64, 64, 1)))
        # known_contact = known_contact[0, 50, 32, 32, 0].assign(1)
        # VG_PUB.publish('aux', known_contact)
        known_contact = self.last_visible_vg['known_occ']
        known_free = self.swept_freespace

        rospy.sleep(2)

        for i in range(100):
            pssnet.grad_step_towards_output(latent, known_contact, known_free)
            self.robot_view.VG_PUB.publish('predicted_occ', pssnet.decode(latent, apply_sigmoid=True))
        return pssnet.decode(latent, apply_sigmoid=True)
