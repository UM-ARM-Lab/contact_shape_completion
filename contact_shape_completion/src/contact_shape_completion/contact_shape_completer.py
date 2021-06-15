from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import rospkg
import rospy
import tensorflow as tf
from colorama import Fore
from sensor_msgs.msg import PointCloud2

import ros_numpy
from contact_shape_completion import contact_tools
from contact_shape_completion.beliefs import ParticleBelief, Particle
from contact_shape_completion.contact_tools import enforce_contact, enforce_contact_ignore_latent_prior, \
    are_chss_satisfied
from contact_shape_completion.kinect_listener import DepthCameraListener
from contact_shape_completion.scenes import Scene, LiveScene, SimulationScene, SceneType
from gpu_voxel_planning_msgs.srv import CompleteShape, CompleteShapeResponse, CompleteShapeRequest, RequestShape, \
    RequestShapeResponse, RequestShapeRequest, ResetShapeCompleterRequest, ResetShapeCompleterResponse, \
    ResetShapeCompleter
from rviz_voxelgrid_visuals import conversions as visual_conversions
from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.utils.tf_utils import add_batch_to_dict, stack_known, compute_quantiles
from shape_completion_training.voxelgrid import conversions
from shape_completion_training.voxelgrid.utils import inflate_voxelgrid

tf.get_logger().setLevel('ERROR')


class ContactShapeCompleter:
    def __init__(self, scene: Scene, trial=None, store_request=False,
                 completion_density=3,
                 method='proposed'):
        self.scene = scene
        self.should_store_request = store_request
        self.completion_density = completion_density
        self.method = method

        self.robot_view = DepthCameraListener(voxelgrid_forward_shift=scene.forward_shift_for_voxelgrid,
                                              object_categories=self.scene.segmented_object_categories,
                                              scale=scene.scale)
        self.model_runner = None
        if trial is not None:
            self.load_network(trial)

        self.complete_shape = rospy.Service("complete_shape", CompleteShape, self.complete_shape_srv)
        self.request_shape = rospy.Service("get_known_world", RequestShape, self.request_known_world_srv)
        self.request_true_shape = rospy.Service("get_true_world", RequestShape, self.request_true_world_srv)
        self.reset_completer = rospy.Service("reset_completer", ResetShapeCompleter, self.reset_completer_srv)
        self.last_visible_vg = None
        self.belief = ParticleBelief()
        self.known_obstacles = None

        self.prev_shape_completion_request = None

    def unsubscribe(self):
        self.complete_shape.shutdown()
        self.request_shape.shutdown()
        self.request_true_shape.shutdown()
        self.reset_completer.shutdown()

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
        print(f"{Fore.RED}Reloading flow for inference (Are you sure this should be happening?){Fore.RESET}")
        self.model_runner.model.flow = ModelRunner(training=False,
                                                   trial_path=self.model_runner.params['flow']).model.flow

    def load_network(self, trial):
        if trial is None:
            print(f"{Fore.RED}Not loading any inference model{Fore.RESET}")
            return
        self.model_runner = ModelRunner(training=False, trial_path=trial)

    def get_visible_vg(self, load=False):
        if self.scene.scene_type == SceneType.LIVE:
            save_name = f"{self.scene.name}_latest_segmented_pts.msg"
            if load:
                self.load_visible_vg(save_name)
            else:
                if self.should_store_request:
                    save_path = self.scene.get_save_path() / save_name
                else:
                    save_path = self.get_debug_save_path() / save_name
                self.last_visible_vg = self.robot_view.get_visible_element(save_file=save_path)

        elif self.scene.scene_type == SceneType.SIMULATION:
            self.last_visible_vg = self.robot_view.voxelize_visible_element(self.scene.get_segmented_points())
        return self.last_visible_vg

    def load_visible_vg(self, filename):
        if self.scene.scene_type == SceneType.LIVE:
            pt_msg = PointCloud2()
            with (self.scene.get_save_path() / filename).open('rb') as f:
                pt_msg.deserialize(f.read())
            self.last_visible_vg = self.robot_view.voxelize_visible_element(pt_msg)
        elif self.scene.scene_type == SceneType.SIMULATION:
            raise RuntimeError("What are you doing loading the visible_vg from a simulation scene?")
        else:
            raise RuntimeError("Unknown scene type")

    @staticmethod
    def get_debug_save_path():
        path = Path(rospkg.RosPack().get_path("contact_shape_completion")) / "debugging/files"
        path.mkdir(exist_ok=True)
        return path

    def request_shape_srv(self, req: RequestShapeRequest):
        pt = self.transform_to_gpuvoxels(self.last_visible_vg['known_occ'])
        return RequestShapeResponse(points=pt)

    def compute_known_occ(self):
        if isinstance(self.scene, LiveScene):
            pts = self.robot_view.point_cloud_creator.unfiltered_pointcloud()
            pts = self.robot_view.transform_pts_to_target(pts, target_frame="gpu_voxel_world")

            # Compute pts exactly as gpu voxels would see the obstacles
            pts = contact_tools.denoise_pointcloud(pts, scale=0.02, origin=[0, 0, 0],
                                                   shape=[256, 256, 256],
                                                   threshold=50)

            box_bounds = contact_tools.BoxBounds(x_lower=0, x_upper=3,
                                                 y_lower=0, y_upper=5,
                                                 z_lower=0, z_upper=1.2)
            pts = contact_tools.remove_points_outside_box(pts, box_bounds)

            self.known_obstacles = visual_conversions.points_to_pointcloud2_msg(pts, frame="gpu_voxel_world")

        elif isinstance(self.scene, SimulationScene):
            self.known_obstacles = self.scene.get_segmented_points()

        return self.known_obstacles

    def request_known_world_srv(self, req: RequestShapeRequest):
        return RequestShapeResponse(points=self.known_obstacles)

    def request_true_world_srv(self, req: RequestShapeRequest):
        pt = self.scene.get_gt()
        return RequestShapeResponse(points=pt)

    def save_request(self, req):
        if self.should_store_request:
            stamp = "{:%Y_%B_%d_%H-%M-%S}".format(datetime.now())
            with (self.scene.get_save_path() / f"req_{stamp}.msg").open('wb') as f:
                req.serialize(f)
            return

        # By default, store the latest request for use with debugging
        with (self.get_debug_save_path() / "latest_request.msg").open('wb') as f:
            req.serialize(f)

    def is_new_request(self, req):
        if req is None:
            return True
        if len(self.belief.particles) == 0:
            return True
        if req == self.prev_shape_completion_request:
            print(f"{Fore.YELLOW}New request is same as previous: Not updating{Fore.RESET}")
            return False
        return True

    def complete_shape_srv(self, req: CompleteShapeRequest):
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

        if self.is_new_request(req):
            self.save_request(req)
            self.update_belief(known_free, chss, req.num_samples)
            self.prev_shape_completion_request = req

        resp = CompleteShapeResponse()
        for p in self.belief.particles:
            if p.successful_projection or self.method == 'baseline_accept_failed_projections':
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

        if self.method == "proposed":
            self.update_belief_proposed(known_free, chss)
        elif self.method == "baseline_ignore_latent_prior":
            self.update_belief_proposed(known_free, chss)  # Ablation done in enforce_contact
        elif self.method == "baseline_accept_failed_projections":
            self.update_belief_proposed(known_free, chss)  # Difference comes when returning message
        elif self.method == "baseline_OOD_prediction":
            self.update_belief_OOD_prediction(known_free, chss)
        elif self.method == "baseline_rejection_sampling":
            self.update_belief_rejection_sampling(known_free, chss)
        else:
            raise RuntimeError(f"Unknown method {self.method}. Cannot update belief")

        for i, p in enumerate(self.belief.particles):
            self.scene.goal_generator.publish_goal(p.goal, marker_id=i)

    def update_belief_proposed(self, known_free, chss):
        # First update current particles (If current particles exist, the prior mean and logvar must have been set
        for particle in self.belief.particles:
            self.scene.goal_generator.clear_goal_markers()
            particle.latent, particle.successful_projection = self.enforce_contact(particle.latent, known_free, chss)
            predicted_occ = self.model_runner.model.decode(particle.latent, apply_sigmoid=True)
            pts = self.transform_to_gpuvoxels(predicted_occ)
            self.robot_view.VG_PUB.publish('predicted_occ', predicted_occ)
            try:
                goal_tsr = self.scene.goal_generator.generate_goal_tsr(pts)
            except RuntimeError as e:
                print(e)
                continue
            particle.goal = goal_tsr
            particle.completion = pts

    def update_belief_OOD_prediction(self, known_free, chss):
        visible_vg = deepcopy(self.last_visible_vg)
        visible_vg['known_free'] += known_free

        if chss is not None:
            gt = self.transform_from_gpuvoxels(self.scene.get_gt())
            gt = inflate_voxelgrid(tf.expand_dims(gt, axis=0))[0, :, :, :, :]
            contact_voxels = tf.reduce_sum(chss, axis=0)
            self.robot_view.VG_PUB.publish('chs', contact_voxels)
            self.robot_view.VG_PUB.publish('gt', gt)
            visible_vg['known_occ'] = np.clip(contact_voxels * gt + visible_vg['known_occ'], 0.0, 1.0)
            self.robot_view.VG_PUB.publish('known_contact', contact_voxels * gt)

        for particle in self.belief.particles:
            self.scene.goal_generator.clear_goal_markers()
            self.robot_view.VG_PUB.publish('known_free', visible_vg['known_free'])
            predicted_occ = self.model_runner.model.call(add_batch_to_dict(visible_vg), apply_sigmoid=True)[
                'predicted_occ']
            pts = self.transform_to_gpuvoxels(predicted_occ)
            self.robot_view.VG_PUB.publish('predicted_occ', predicted_occ)
            try:
                goal_tsr = self.scene.goal_generator.generate_goal_tsr(pts)
            except RuntimeError as e:
                print(e)
                continue
            particle.goal = goal_tsr
            particle.completion = pts

    def update_belief_rejection_sampling(self, known_free, chss):
        for particle in self.belief.particles:
            self.scene.goal_generator.clear_goal_markers()
            self.robot_view.VG_PUB.publish('known_free', known_free)
            predicted_occ = self.model_runner.model.call(add_batch_to_dict(self.last_visible_vg), apply_sigmoid=True)[
                'predicted_occ']
            pts = self.transform_to_gpuvoxels(predicted_occ)
            self.robot_view.VG_PUB.publish('predicted_occ', predicted_occ)
            try:
                goal_tsr = self.scene.goal_generator.generate_goal_tsr(pts)
            except RuntimeError as e:
                print(e)
                continue
            particle.goal = goal_tsr
            particle.completion = pts

            particle.successful_projection = True
            if tf.reduce_sum(known_free * predicted_occ) >= 1.0:
                particle.successful_projection = False
            if not are_chss_satisfied(predicted_occ, chss):
                particle.successful_projection = False

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
                                                        density_factor=self.completion_density)
        # pt = conversions.voxelgrid_to_pointcloud(vg, scale=self.robot_view.scale,
        #                                          origin=self.robot_view.origin)

        msg = self.robot_view.transform_pts_to_target(msg, target_frame="gpu_voxel_world")
        return msg

    def do_some_completions_debug(self):
        known_free = np.zeros((64, 64, 64, 1), dtype=np.float32)
        self.update_belief(known_free, None, 10)
        rospy.sleep(5)
        self.update_belief(known_free, None, 10)

    def enforce_contact(self, latent, known_free, chss):
        if self.method == "proposed" or \
                self.method == 'baseline_accept_failed_projections':
            return enforce_contact(latent, known_free, chss, self.model_runner.model,
                                   self.belief, self.robot_view.VG_PUB)
        elif self.method == "baseline_ignore_latent_prior":
            return enforce_contact_ignore_latent_prior(latent, known_free, chss, self.model_runner.model,
                                                       self.belief, self.robot_view.VG_PUB)
        else:
            raise RuntimeError(f"Not known how to enforce_contact for method {self.method}")
