import ros_numpy
from contact_shape_completion.kinect_listener import DepthCameraListener
from gpu_voxel_planning_msgs.srv import CompleteShape, CompleteShapeResponse, CompleteShapeRequest, RequestShape, \
    RequestShapeResponse, RequestShapeRequest
from gpu_voxel_planning_msgs.msg import JointConfig
from shape_completion_training.voxelgrid import conversions
import rospy
from sensor_msgs.msg import PointCloud2
from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.utils.tf_utils import add_batch_to_dict, stack_known
import tensorflow as tf
import numpy as np
from contact_shape_completion.goal_generator import GoalGenerator

from rviz_voxelgrid_visuals import conversions as visual_conversions

tf.get_logger().setLevel('ERROR')


class ContactShapeCompleter:
    def __init__(self, trial=None, goal_generator=None):
        self.goal_generator = goal_generator  # type GoalGenerator
        self.robot_view = DepthCameraListener()
        self.model_runner = None
        if trial is not None:
            self.load_network(trial)
        # self.model = None

        self.complete_shape = rospy.Service("complete_shape", CompleteShape, self.complete_shape_srv)
        self.request_shape = rospy.Service("get_shape", RequestShape, self.request_shape_srv)
        self.new_free_sub = rospy.Subscriber("swept_freespace_pointcloud", PointCloud2,
                                             self.new_swept_freespace_callback)
        self.pointcloud_repub = rospy.Publisher("swept_volume_republisher", PointCloud2, queue_size=10)
        self.last_visible_vg = None
        self.swept_freespace = tf.zeros((1, 64, 64, 64, 1))

    def reload_flow(self):
        """
        For an unknown reason, I need to reload the flow whenever calling pssnet within a thread (e.g. ros service
        callback).

        """
        if not self.model_runner.params['use_flow_during_inference']:
            return
        self.model_runner.model.flow = ModelRunner(training=False,
                                                   trial_path=self.model_runner.params['flow']).model.flow

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

        self.reload_flow()

        # self.do_some_completions_debug()

        # inference = self.model_runner.model(add_batch_to_dict(self.last_visible_vg))
        pt = self.transform_to_gpuvoxels(self.last_visible_vg['known_occ'])

        return RequestShapeResponse(points=pt)

    def complete_shape_srv(self, req: CompleteShapeRequest):
        print(f"{req.num_samples} shape completions requested")

        # print(req)
        if self.model_runner is None:
            raise AttributeError("Model must be loaded before inferring completion")
        self.reload_flow()

        if req.num_samples <= 0:
            raise ValueError(f"{req.num_samples} samples requested. Probably a mistake")

        # self.do_some_completions_debug()

        resp = CompleteShapeResponse()

        # TODO: Handle case where a scenario has no valid goal
        while len(resp.sampled_completions) < req.num_samples:
            # for _ in range(req.num_samples):
            inference = self.model_runner.model(add_batch_to_dict(self.last_visible_vg))
            pts = self.transform_to_gpuvoxels(inference['predicted_occ'])
            self.robot_view.VG_PUB.publish('predicted_occ', inference['predicted_occ'])
            # goal_config = self.goal_generator.generate_goal(pts)
            # if goal_config is None:
            #     continue
            try:
                goal_tsr = self.goal_generator.generate_goal_tsr(pts)
            except RuntimeError as e:
                print(e)
                continue


            resp.sampled_completions.append(pts)
            resp.goal_tsrs.append(goal_tsr)
            # resp.goal_configs.append(JointConfig(joint_values=goal_config))
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

    def do_some_completions_debug(self):

        # mr2 = ModelRunner(trial_path=self.model_runner.trial_path, training=False)
        # pssnet = mr2.model
        # ModelRunner(trial_path=self.model_runner.trial_path, training=False)
        # pssnet = self.model_runner.model  # type PSSNet
        # mean, logvar = pssnet.encode(stack_known(add_batch_to_dict(self.last_visible_vg)))

        # pssnet.hparams['use_flow_during_inference']
        # output = pssnet.decode(mean, apply_sigmoid=True)
        # print(tf.reduce_sum(mean).numpy())
        # print(tf.reduce_mean(output).numpy())
        # latent = pssnet.sample_latent(add_batch_to_dict(self.last_visible_vg))
        # output = pssnet.decode(latent)
        # flow = pssnet.apply_flow_to_latent_box(mean)
        # print(tf.reduce_sum(flow).numpy())
        #
        # flow_in = np.array([np.float32(i) for i in range(1, 25)])
        # flow_in = np.expand_dims(flow_in, axis=0) / 100
        #
        # flow = pssnet.flow  # Type RealNVP
        # # flow.
        #
        # flow_tmp = flow_in

        # print(f"input: {tf.reduce_sum(flow_tmp).numpy()}")
        # for bijector in reversed(flow.bijector.bijectors):
        #     flow_tmp = bijector(flow_tmp)
        #     print(f"{bijector.name:<20} {tf.reduce_sum(flow_tmp).numpy()}")

        # flow_out = pssnet.flow.bijector.forward(flow_in, training=False)
        # print(f'Calling bijector forward: {tf.reduce_sum(flow_out).numpy()}')

        # self.robot_view.VG_PUB.publish('predicted_occ', output)

        for _ in range(30):
            latent = tf.Variable(self.model_runner.model.sample_latent(add_batch_to_dict(self.last_visible_vg)))
            predicted_occ = self.model_runner.model.decode(latent, apply_sigmoid=True)
            self.robot_view.VG_PUB.publish('predicted_occ', predicted_occ)

    def infer_completion(self):
        if self.model_runner is None:
            raise AttributeError("Model must be loaded before inferring completion")

        inference = self.model_runner.model(add_batch_to_dict(self.last_visible_vg))

        # inference['predicted_occ'] -= self.swept_freespace

        # TODO: I currently can only use this sampling behavior in debug mode by setting a breakpoint
        latent = tf.Variable(self.model_runner.model.sample_latent(add_batch_to_dict(self.last_visible_vg)))
        predicted_occ = self.model_runner.model.decode(latent, apply_sigmoid=True)
        self.robot_view.VG_PUB.publish('predicted_occ', predicted_occ)

        # self.do_some_completions_debug()

        # self.enforce_contact(latent)

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
