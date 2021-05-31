from pathlib import Path

import numpy as np
import rospkg
import rospy
import tensorflow as tf
from colorama import Fore
from deprecated import deprecated
from sensor_msgs.msg import PointCloud2

import ros_numpy
from contact_shape_completion import contact_tools
from contact_shape_completion.kinect_listener import DepthCameraListener
from contact_shape_completion.simulation_ground_truth_scenes import scene1_gt
from gpu_voxel_planning_msgs.srv import CompleteShape, CompleteShapeResponse, CompleteShapeRequest, RequestShape, \
    RequestShapeResponse, RequestShapeRequest, ResetShapeCompleterRequest, ResetShapeCompleterResponse, \
    ResetShapeCompleter
from rviz_voxelgrid_visuals import conversions as visual_conversions
from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.utils.tf_utils import add_batch_to_dict, stack_known, log_normal_pdf, compute_quantiles
from shape_completion_training.voxelgrid import conversions

tf.get_logger().setLevel('ERROR')

GRADIENT_UPDATE_ITERATION_LIMIT = 100
KNOWN_FREE_LIMIT = 0.4
KNOWN_OCC_LIMIT = 0.5


class ParticleBelief:
    def __init__(self):
        self.latent_prior_mean = None
        self.latent_prior_logvar = None
        self.particles = []
        self.quantiles_log_pdf = None

    def reset(self):
        self.latent_prior_mean = None
        self.latent_prior_logvar = None
        self.particles = []

    def get_quantile(self, log_pdf):
        return tf.where(self.quantiles_log_pdf > log_pdf)[0, 0].numpy()


class Particle:
    def __init__(self):
        self.sampled_latent = None
        self.latent = None
        self.goal = None
        self.completion = None
        self.associated_chs_inds = []


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
        self.belief = ParticleBelief()
        self.known_obstacles = None

    def reset_completer_srv(self, req: ResetShapeCompleterRequest):
        self.belief.reset()
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
        save_path = self.get_wip_save_path() / "latest_segmented_pts.msg"
        self.last_visible_vg = self.robot_view.get_visible_element(save_file=save_path)
        # self.save_last_visible_vg()
        return self.last_visible_vg

    def load_visible_vg(self, filename='latest_segmented_pts.msg'):
        pt_msg = PointCloud2()
        with (self.get_wip_save_path() / filename).open('rb') as f:
            pt_msg.deserialize(f.read())
        self.last_visible_vg = self.robot_view.get_visible_element()

    @staticmethod
    def get_wip_save_path():
        path = Path(rospkg.RosPack().get_path("contact_shape_completion")) / "debugging/files"
        path.mkdir(exist_ok=True)
        return path

    @deprecated
    def save_last_visible_vg(self):
        path = self.get_wip_save_path() / "wip_visible_vg.npz"

        with path.open('wb') as f:
            np.savez_compressed(f, **self.last_visible_vg)

    @deprecated
    def load_last_visible_vg(self, filename="wip_visible_vg.npz"):
        path = self.get_wip_save_path() / filename
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
        # pt = self.transform_to_gpuvoxels(self.last_visible_vg['known_occ'])
        pt = scene1_gt()
        return RequestShapeResponse(points=pt)

    def complete_shape_srv(self, req: CompleteShapeRequest):
        with (self.get_wip_save_path() / "wip_req").open('wb') as f:
            req.serialize(f)

        print(f"{Fore.GREEN}{req.num_samples} shape completions requested with {len(req.chss)} chss{Fore.RESET}")

        if self.model_runner is None:
            raise AttributeError("Model must be loaded before inferring completion")

        if req.num_samples <= 0:
            raise ValueError(f"{req.num_samples} samples requested. Probably a mistake")

        known_free = self.transform_from_gpuvoxels(req.known_free)
        if len(req.chss) == 0:
            chss = None
        else:
            chss = tf.concat([tf.expand_dims(self.transform_from_gpuvoxels(chs), axis=0) for chs in req.chss], axis=0)
            if tf.reduce_max(known_free * chss) > 0:
                raise RuntimeError("Known free overlaps with CHSs")

        self.update_belief(known_free, chss, req.num_samples)
        resp = CompleteShapeResponse()
        for p in self.belief.particles:
            resp.sampled_completions.append(p.completion)
            resp.goal_tsrs.append(p.goal)
        return resp

    def initialize_belief(self, num_particles):
        pssnet = self.model_runner.model
        mean, logvar = pssnet.encode(stack_known(add_batch_to_dict(self.last_visible_vg)))
        self.belief.latent_prior_mean = mean
        self.belief.latent_prior_logvar = logvar
        self.belief.quantiles_log_pdf = compute_quantiles(mean, logvar, num_quantiles=100, num_samples=1000)

        for _ in range(num_particles):
            p = Particle()
            latent = pssnet.sample_latent_from_mean_and_logvar(mean, logvar)
            p.latent = tf.Variable(latent)
            p.sampled_latent = latent
            self.belief.particles.append(p)

    def update_belief(self, known_free, chss, num_particles):
        self.reload_flow()
        pssnet = self.model_runner.model

        if len(self.belief.particles) > num_particles:
            raise RuntimeError("Unexpected situation - we have more particles than requested")

        if self.belief.latent_prior_mean is None:
            self.initialize_belief(num_particles)

        if len(self.belief.particles) != num_particles:
            raise RuntimeError("Unexpected situation - number of particles does not match request")

        # TODO: This is a debugging script only
        # self.debug_repeated_sampling(self.belief, known_free, chss)
        # if chss is not None:
        #     self.debug_repeated_sampling(self.belief, known_free, chss)

        # First update current particles (If current particles exist, the prior mean and logvar must have been set
        for particle in self.belief.particles:
            self.goal_generator.clear_goal_markers()
            particle.latent = self.enforce_contact(particle.latent, known_free, chss)
            predicted_occ = self.model_runner.model.decode(particle.latent, apply_sigmoid=True)
            pts = self.transform_to_gpuvoxels(predicted_occ)
            self.robot_view.VG_PUB.publish('predicted_occ', predicted_occ)
            try:
                goal_tsr = self.goal_generator.generate_goal_tsr(pts)
            except RuntimeError as e:
                print(e)
                continue
            particle.goal = goal_tsr
            particle.completion = pts

        for i, p in enumerate(self.belief.particles):
            self.goal_generator.publish_goal(p.goal, marker_id=i)

    def debug_repeated_sampling(self, bel: ParticleBelief, known_free, chss):
        pssnet = self.model_runner.model
        latent = tf.Variable(pssnet.sample_latent_from_mean_and_logvar(bel.latent_prior_mean, bel.latent_prior_logvar))
        pred_occ = pssnet.decode(latent, apply_sigmoid=True)
        # known_contact = contact_tools.get_assumed_occ(pred_occ, chss)
        self.robot_view.VG_PUB.publish('predicted_occ', pred_occ)
        # self.robot_view.VG_PUB.publish('known_contact', known_contact)
        # latent = self.enforce_contact(latent, known_free, chss)

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

        known_free = np.zeros((64, 64, 64, 1), dtype=np.float32)
        self.update_belief(known_free, None, 10)
        rospy.sleep(5)
        self.update_belief(known_free, None, 10)

    def enforce_contact(self, latent, known_free, chss):
        self.robot_view.VG_PUB.publish("aux", 0 * known_free)
        pssnet = self.model_runner.model
        prior_mean, prior_logvar = pssnet.encode(stack_known(add_batch_to_dict(self.last_visible_vg)))
        p = log_normal_pdf(latent, prior_mean, prior_logvar)
        self.robot_view.VG_PUB.publish('predicted_occ', pssnet.decode(latent, apply_sigmoid=True))
        pred_occ = pssnet.decode(latent, apply_sigmoid=True)
        known_contact = contact_tools.get_assumed_occ(pred_occ, chss)
        any_chs = tf.reduce_max(chss, axis=-1, keepdims=True)
        self.robot_view.VG_PUB.publish('known_free', known_free)
        if chss is not None:
            self.robot_view.VG_PUB.publish('chs', chss)

        print()
        log_pdf = log_normal_pdf(latent, prior_mean, prior_logvar)
        quantile = self.belief.get_quantile(log_pdf)
        print(f"Before optimization logprob: {log_pdf} ({quantile} quantile)")

        prev_loss = 0.0
        for i in range(GRADIENT_UPDATE_ITERATION_LIMIT):
            single_free = contact_tools.get_most_wrong_free(pred_occ, known_free)

            # loss = pssnet.grad_step_towards_output(latent, known_contact, known_free)
            loss = pssnet.grad_step_towards_output(latent, known_contact, single_free)
            print('\rloss: {}'.format(loss), end='')
            pred_occ = pssnet.decode(latent, apply_sigmoid=True)
            known_contact = contact_tools.get_assumed_occ(pred_occ, chss)
            self.robot_view.VG_PUB.publish('predicted_occ', pred_occ)
            self.robot_view.VG_PUB.publish('known_contact', known_contact)

            if loss == prev_loss:
                print("\tNo progress made. Accepting shape as is")
                break
            prev_loss = loss
            if tf.math.is_nan(loss):
                print("\tLoss is nan. There is a problem I am not addressing")
                break

            if np.max(pred_occ * known_free) <= KNOWN_FREE_LIMIT and \
                    tf.reduce_min(tf.boolean_mask(pred_occ, known_contact)) >= KNOWN_OCC_LIMIT:
                print(
                    f"\tAll known free have less that {KNOWN_FREE_LIMIT} prob occupancy, "
                    f"and chss have value > {KNOWN_OCC_LIMIT}")
                break
        else:
            print()
            if np.max(pred_occ * known_free) > KNOWN_FREE_LIMIT:
                print("Optimization did not satisfy known freespace: ")
                self.robot_view.VG_PUB.publish("aux", pred_occ * known_free)
            if tf.reduce_min(tf.boolean_mask(pred_occ, known_contact)) < KNOWN_OCC_LIMIT:
                print("Optimization did not satisfy assumed occ: ")
                self.robot_view.VG_PUB.publish("aux", (1 - pred_occ) * known_contact)
            print('\tWarning, enforcing contact terminated due to max iterations, not actually satisfying contact')

        log_pdf = log_normal_pdf(latent, prior_mean, prior_logvar)
        quantile = self.belief.get_quantile(log_pdf)
        print(f"Latent logprob {log_pdf} ({quantile} quantile)")
        return latent
