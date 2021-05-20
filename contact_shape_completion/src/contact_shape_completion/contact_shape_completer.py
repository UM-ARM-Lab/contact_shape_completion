import rospkg

import ros_numpy
from contact_shape_completion import contact_tools
from contact_shape_completion.kinect_listener import DepthCameraListener
from gpu_voxel_planning_msgs.srv import CompleteShape, CompleteShapeResponse, CompleteShapeRequest, RequestShape, \
    RequestShapeResponse, RequestShapeRequest, ResetShapeCompleterRequest, ResetShapeCompleterResponse, \
    ResetShapeCompleter
from shape_completion_training.voxelgrid import conversions
import rospy
from sensor_msgs.msg import PointCloud2
from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.utils.tf_utils import add_batch_to_dict, stack_known, log_normal_pdf
import tensorflow as tf
import numpy as np
from pathlib import Path
from colorama import Fore
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

        self.complete_shape = rospy.Service("complete_shape", CompleteShape, self.complete_shape_srv)
        self.request_shape = rospy.Service("get_known_world", RequestShape, self.request_known_world_srv)
        self.request_shape = rospy.Service("get_true_world", RequestShape, self.request_true_world_srv)
        self.reset_completer = rospy.Service("reset_completer", ResetShapeCompleter, self.reset_completer_srv)
        self.new_free_sub = rospy.Subscriber("swept_freespace_pointcloud", PointCloud2,
                                             self.new_swept_freespace_callback)
        self.pointcloud_repub = rospy.Publisher("swept_volume_republisher", PointCloud2, queue_size=10)
        self.last_visible_vg = None
        self.swept_freespace = tf.zeros((1, 64, 64, 64, 1))
        self.sampled_latent_particles = []
        self.known_obstacles = None

    def reset_completer_srv(self, req: ResetShapeCompleterRequest):
        self.sampled_latent_particles = []
        return ResetShapeCompleterResponse(reset_complete=True)

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
        if trial is None:
            print("Not loading any inference model")
            return
        self.model_runner = ModelRunner(training=False, trial_path=trial)

    def get_visible_vg(self):
        self.last_visible_vg = self.robot_view.get_visible_element()

    @staticmethod
    def get_wip_save_path():
        path = Path(rospkg.RosPack().get_path("contact_shape_completion")) / "tests/files"
        path.mkdir(exist_ok=True)
        return path / "visible_vg.npz"

    def save_last_visible_vg(self):
        path = self.get_wip_save_path()

        with path.open('wb') as f:
            np.savez_compressed(f, **self.last_visible_vg)

    def load_last_visible_vg(self):
        path = self.get_wip_save_path()
        self.last_visible_vg = np.load(path.as_posix())

    def request_shape_srv(self, req: RequestShapeRequest):
        pt = self.transform_to_gpuvoxels(self.last_visible_vg['known_occ'])
        return RequestShapeResponse(points=pt)

    def compute_known_occ(self):
        pts = self.robot_view.point_cloud_creator.unfiltered_pointcloud()
        pts = self.robot_view.transform_pts_to_target(pts, target_frame="gpu_voxel_world")
        pts = contact_tools.denoise_pointcloud(pts, scale=0.02, origin=[0, 0, 0], shape=[256, 256, 256], threshold=100)
        self.known_obstacles = visual_conversions.points_to_pointcloud2_msg(pts, frame="gpu_voxel_world")

        return self.known_obstacles

    def request_known_world_srv(self, req: RequestShapeRequest):
        # pt = self.robot_view.point_cloud_creator.unfiltered_pointcloud()
        # pt = self.robot_view.transform_pts_to_target(pt, target_frame="gpu_voxel_world")
        return RequestShapeResponse(points=self.known_obstacles)

    def request_true_world_srv(self, req: RequestShapeRequest):
        pt = self.transform_to_gpuvoxels(self.last_visible_vg['known_occ'])
        return RequestShapeResponse(points=pt)

    def complete_shape_srv(self, req: CompleteShapeRequest):
        print(f"{Fore.GREEN}{req.num_samples} shape completions requested with {len(req.chss)} chss{Fore.RESET}")

        if self.model_runner is None:
            raise AttributeError("Model must be loaded before inferring completion")
        self.reload_flow()

        if req.num_samples <= 0:
            raise ValueError(f"{req.num_samples} samples requested. Probably a mistake")

        # self.do_some_completions_debug()
        known_free = self.transform_from_gpuvoxels(req.known_free)
        # TODO: Pass chss
        if len(req.chss) == 0:
            chss = None
        else:
            chss = tf.concat([tf.expand_dims(self.transform_from_gpuvoxels(chs), axis=0) for chs in req.chss], axis=0)

        resp = CompleteShapeResponse()

        prev_latents = self.sampled_latent_particles
        self.sampled_latent_particles = []

        # TODO: Handle case where a scenario has no valid goal
        while len(resp.sampled_completions) < req.num_samples:
            if len(prev_latents) > 0:
                latent = prev_latents.pop()
            else:
                latent = tf.Variable(self.model_runner.model.sample_latent(add_batch_to_dict(self.last_visible_vg)))

            self.goal_generator.clear_goal_markers()
            latent = self.enforce_contact(latent, known_free, chss)
            predicted_occ = self.model_runner.model.decode(latent, apply_sigmoid=True)
            pts = self.transform_to_gpuvoxels(predicted_occ)

            self.robot_view.VG_PUB.publish('predicted_occ', predicted_occ)

            try:
                goal_tsr = self.goal_generator.generate_goal_tsr(pts)
            except RuntimeError as e:
                print(e)
                continue
            self.sampled_latent_particles.append(latent)
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
        # pt_cloud = conversions.voxelgrid_to_pointcloud(vg, scale=self.robot_view.scale,
        #                                                origin=self.robot_view.origin)

        # TODO: It is odd that I use visual_conversions here, since I used conversions (not visual, different package
        #  of mine) get the pointcloud in the first place. However, visual_conversions has this nice function which
        #  densifies the points
        msg = visual_conversions.vox_to_pointcloud2_msg(vg, frame=self.robot_view.target_frame,
                                                        scale=self.robot_view.scale,
                                                        origin=-self.robot_view.origin / self.robot_view.scale,
                                                        density_factor=3)
        # pt = conversions.voxelgrid_to_pointcloud(vg, scale=self.robot_view.scale,
        #                                          origin=self.robot_view.origin)

        msg = self.robot_view.transform_pts_to_target(msg, target_frame="gpu_voxel_world")
        return msg

    # TODO: This function is not necessary for the algorithm, just helps debug
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

        latent = tf.Variable(self.model_runner.model.sample_latent(add_batch_to_dict(self.last_visible_vg)))
        predicted_occ = self.model_runner.model.decode(latent, apply_sigmoid=True)
        self.robot_view.VG_PUB.publish('predicted_occ', predicted_occ)

        return inference

    def wip_enforce_contact(self, latent):
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
            pred_occ = pssnet.decode(latent, apply_sigmoid=True)
            self.robot_view.VG_PUB.publish('predicted_occ', pred_occ)

            if np.max(pred_occ * known_free) <= 0.5:
                break

        return latent

    def enforce_contact(self, latent, known_free, chss):
        pssnet = self.model_runner.model
        prior_mean, prior_logvar = pssnet.encode(stack_known(add_batch_to_dict(self.last_visible_vg)))
        p = log_normal_pdf(latent, prior_mean, prior_logvar)
        self.robot_view.VG_PUB.publish('predicted_occ', pssnet.decode(latent, apply_sigmoid=True))
        pred_occ = pssnet.decode(latent, apply_sigmoid=True)
        known_contact = contact_tools.get_assumed_occ(pred_occ, chss)
        self.robot_view.VG_PUB.publish('known_free', known_free)

        prev_loss = 0.0
        for i in range(500):
            loss = pssnet.grad_step_towards_output(latent, known_contact, known_free)
            print('\rloss: {}'.format(loss), end='')
            pred_occ = pssnet.decode(latent, apply_sigmoid=True)
            known_contact = contact_tools.get_assumed_occ(pred_occ, chss)
            self.robot_view.VG_PUB.publish('predicted_occ', pred_occ)
            self.robot_view.VG_PUB.publish('chs', known_contact)

            if loss == prev_loss:
                print("\tNo progress made. Accepting shape as is")
                break
            prev_loss = loss
            if tf.math.is_nan(loss):
                print("\tLoss is nan. There is a problem I am not addressing")
                break

            if np.max(pred_occ * known_free) <= 0.2 and tf.reduce_min(tf.boolean_mask(pred_occ, known_contact)) >= 0.5:
                print("\tAll known free have less that 0.2 prob occupancy, and chss have value > 0.5")
                break
        else:
            print('\tWarning, enforcing contact terminated due to max iterations, not actually satisfying contact')

        print(f"Latent now has logprob {log_normal_pdf(latent, prior_mean, prior_logvar)}")
        return latent
