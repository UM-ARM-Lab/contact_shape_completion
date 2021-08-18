import random
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
from contact_shape_completion.beliefs import ParticleBelief, Particle, MultiObjectParticleBelief
from contact_shape_completion.contact_tools import enforce_contact, enforce_contact_ignore_latent_prior, \
    are_chss_satisfied
from contact_shape_completion.kinect_listener import DepthCameraListener
from contact_shape_completion.scenes import Scene, LiveScene, SimulationScene, SceneType, combine_pointcloud2s
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

        self.robot_views = []
        for i in range(scene.num_objects):
            self.robot_views.append(DepthCameraListener(voxelgrid_forward_shift=scene.forward_shift_for_voxelgrid,
                                                        object_categories=scene.segmented_object_categories[i],
                                                        scale=scene.depth_camera_listener_scale,
                                                        scene_type=scene.scene_type))
        self.model_runner = None
        if trial is not None:
            self.load_network(trial)

        self.complete_shape = rospy.Service("complete_shape", CompleteShape, self.complete_shape_srv)
        self.request_shape = rospy.Service("get_known_world", RequestShape, self.request_known_world_srv)
        self.request_true_shape = rospy.Service("get_true_world", RequestShape, self.request_true_world_srv)
        self.reset_completer = rospy.Service("reset_completer", ResetShapeCompleter, self.reset_completer_srv)
        # self.last_visible_vgs = None
        self.belief = MultiObjectParticleBelief()
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
            #TODO Update for multiobject
            save_name = f"{self.scene.name}_latest_segmented_pts.msg"
            if load:
                self.load_visible_vg(save_name)
            else:
                # raise RuntimeError("Live scene not updated after multiobject refactor")

                for i in range(self.scene.num_objects):
                    save_name = f"{self.scene.name}_latest_segmented_pts_{i}.msg"
                    if self.should_store_request:
                        save_path = self.scene.get_save_path() / save_name
                    else:
                        save_path = self.get_debug_save_path() / save_name
                    self.robot_views[i].get_visible_element(save_file=save_path)


            #     self.last_visible_vg = self.robot_view.get_visible_element(save_file=save_path)

        elif self.scene.scene_type == SceneType.SIMULATION:
            segmented_objects = self.scene.get_segmented_points()
            if len(segmented_objects) != len(self.robot_views):
                raise RuntimeError("Different number of segemented objects and robot views")
            for view, pts in zip(self.robot_views, segmented_objects):
                view.voxelize_visible_element(pts)

            # self.last_visible_vg = self.robot_view.voxelize_visible_element(self.scene.get_segmented_points())
        return [view.last_visible for view in self.robot_views]

    def load_visible_vg(self, filename):
        if self.scene.scene_type == SceneType.LIVE:
            # raise RuntimeError("Live scene not updated after multiobject refactor")
            try:
                pt_msg = PointCloud2()
                with (self.scene.get_save_path() / filename).open('rb') as f:
                    pt_msg.deserialize(f.read())
                if len(self.robot_views) != 1:
                    raise RuntimeError("Live scene with multiple objects, yet single latest segmented point found")
                self.robot_views[0].voxelize_visible_element(pt_msg)
            except FileNotFoundError as e:
                print("Single latest segmeented points not found. You are probably doing this after the refactor")

            for i in range(self.scene.num_objects):
                filename = f"{self.scene.name}_latest_segmented_pts_{i}.msg"
                pt_msg = PointCloud2()
                with (self.scene.get_save_path() / filename).open('rb') as f:
                    pt_msg.deserialize(f.read())
                self.robot_views[i].voxelize_visible_element(pt_msg)

                # self.robot_views[i].get_visible_element(save_file=save_path)


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
        pts = [self.transform_from_gpuvoxels(view=view, pt_msg=view.last_visible) for view in self.robot_views]
        return RequestShapeResponse(points=combine_pointcloud2s(pts))

    def compute_known_occ(self):
        if isinstance(self.scene, LiveScene):
            self.known_obstacles = []
            # raise RuntimeError("Live scene not updated after multiobject refactor")
            for i in range(self.scene.num_objects):
                view = self.robot_views[i]
                pts = view.point_cloud_creator.unfiltered_pointcloud()
                pts = view.transform_pts_to_target(pts, target_frame="gpu_voxel_world")

                # Compute pts exactly as gpu voxels would see the obstacles
                pts = contact_tools.denoise_pointcloud(pts, scale=0.02, origin=[0, 0, 0],
                                                       shape=[256, 256, 256],
                                                       threshold=50)

                box_bounds = contact_tools.BoxBounds(x_lower=0, x_upper=3,
                                                     y_lower=0, y_upper=5,
                                                     z_lower=0, z_upper=1.2)
                pts = contact_tools.remove_points_outside_box(pts, box_bounds)
                # all_pts.append(pts)
                self.known_obstacles.append(visual_conversions.points_to_pointcloud2_msg(pts,
                                                                                         frame="gpu_voxel_world"))

        elif isinstance(self.scene, SimulationScene):
            self.known_obstacles = self.scene.get_segmented_points()

        return self.known_obstacles

    def request_known_world_srv(self, req: RequestShapeRequest):
        return combine_pointcloud2s(self.known_obstacles)

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
        if len(self.belief.particle_beliefs) == 0:
            return True
        for single_object_belief in self.belief.particle_beliefs:
            if len(single_object_belief.particles) == 0:
                return True
        if req == self.prev_shape_completion_request:
            print(f"{Fore.YELLOW}New request is same as previous: Not updating{Fore.RESET}")
            return False
        return True

    def update_possible_chs_assignments(self, req):

        # If there are no new chss, no need to update
        if all([len(req.chss) == len(p.chs_possible) for p in self.belief.particle_beliefs]):
            return False

        if any([len(req.chss) != (len(p.chs_possible) + 1) for p in self.belief.particle_beliefs]):
            print(f"{len(req.chss)}")
            print(f"{[len(p.chs_possible) for p in self.belief.particle_beliefs]}")
            raise RuntimeError("Unexpected case, there is a new chs but the length of the particle beliefs are "
                               "unexpected")

        # Only one particle belief, so the new chs must be assigned to it
        if len(self.belief.particle_beliefs) == 1:
            self.belief.particle_beliefs[0].chs_possible.append(True)
            return True

        # Initially, assume the new chs could be assigned to any object, then prune
        for object_ind, object_bel in enumerate(self.belief.particle_beliefs):
            object_bel.chs_possible.append(True)
            robot_view = self.robot_views[object_ind]

            latest_chs = self.transform_from_gpuvoxels(robot_view, req.chss[-1])

            # If the CHS has no points in the object's voxelgrid, assignment is not possible
            if np.sum(latest_chs) == 0:
                object_bel.chs_possible[-1] = False
                continue

            # If all projections fail, object assignment is not possible
            known_free = self.transform_from_gpuvoxels(robot_view, req.known_free)

            # TODO: I don't actually want to update here, I just want to see if updates are possible
            # Maybe I could deepcopy the belief, but I don't know if that would work with tensorflow latent vectors
            if self.is_projection_possible(object_ind, known_free, tf.expand_dims(latest_chs, axis=0),
                                               req.num_samples):
                print(f'{Fore.CYAN}Object {object_ind} can be projection to new CHS{Fore.RESET}')
            else:
                print(f'{Fore.RED}Object {object_ind} can NOT be projection to new CHS{Fore.RESET}')
                object_bel.chs_possible[-1] = False
        return True

    def is_projection_possible(self, obj_index, known_free, chss, num_samples):
        # First update current particles (If current particles exist, the prior mean and logvar must have been set
        object_bel = deepcopy(self.belief.particle_beliefs[obj_index])
        for particle_num, particle in enumerate(object_bel.particles):
            self.scene.goal_generator.clear_goal_markers()
            _, successful_projection = self.enforce_contact(particle.latent, known_free, chss, obj_index, verbose=False)
            if successful_projection:
                return True

        return False

    def update_chs_assignments(self, req):
        if not self.update_possible_chs_assignments(req):
            return
        possible_new_assignments = [i for i, p in enumerate(self.belief.particle_beliefs) if p.chs_possible[-1]]
        for sampled_assignment in self.belief.sampled_assignments:
            if len(possible_new_assignments) == 0:
                sampled_assignment.append(-1)
            else:
                sampled_assignment.append(random.choice(possible_new_assignments))

    def get_assigned_chss(self, obj_index, req, particle_num):
        robot_view = self.robot_views[obj_index]
        if self.method == "proposed" or self.scene.num_objects == 1:
            assigned_chs_list = [tf.expand_dims(self.transform_from_gpuvoxels(robot_view, chs), axis=0)
                                 for i, chs in enumerate(req.chss)
                                 if self.belief.sampled_assignments[particle_num][i] == obj_index]
            if len(assigned_chs_list) == 0:
                return None
            return tf.concat(assigned_chs_list, axis=0)
        elif self.method == "assign_all_CHS":
            assigned_chs_list = [tf.expand_dims(self.transform_from_gpuvoxels(robot_view, chs), axis=0)
                                 for i, chs in enumerate(req.chss)
                                 if self.belief.particle_beliefs[obj_index].chs_possible[i]]
            if len(assigned_chs_list) == 0:
                return None
            return tf.concat(assigned_chs_list, axis=0)
        else:
            raise RuntimeError(f"Not known how to assign CHSs to {self.scene.num_objects} using method {self.method}")

    def complete_shape_srv(self, req: CompleteShapeRequest):
        print(f"{Fore.GREEN}{req.num_samples} shape completions requested with {len(req.chss)} chss{Fore.RESET}")

        if self.model_runner is None:
            raise AttributeError("Model must be loaded before inferring completion")

        if req.num_samples <= 0:
            raise ValueError(f"{req.num_samples} samples requested. Probably a mistake")

        is_new_request = self.is_new_request(req)
        if is_new_request:
            self.save_request(req)
            self.prev_shape_completion_request = req

        self.update_chs_assignments(req)

        for obj_index in range(len(self.robot_views)):
            if len(self.belief.particle_beliefs) == 0:
                self.initialize_belief(req.num_samples)

            bel = self.belief.particle_beliefs[obj_index]
            robot_view = self.robot_views[obj_index]
            known_free = self.transform_from_gpuvoxels(robot_view, req.known_free)

            if not is_new_request:
                continue

            self.update_belief(obj_index, known_free, req, req.num_samples)

        resp = CompleteShapeResponse()


        # TODO: Figure out goals from multiple particles. Currently, always assuming first object is the goal
        for particle_num in range(req.num_samples):
            full_scene_particle = [p.particles[particle_num] for p in self.belief.particle_beliefs]
            if all([p.successful_projection for p in full_scene_particle]) or \
                    self.method == 'baseline_accept_failed_projections':
                resp.sampled_completions.append(combine_pointcloud2s([p.completion for p in full_scene_particle]))
                resp.goal_tsrs.append(full_scene_particle[0].goal)  # TODO: This hardcoded uses the first object as goal

        return resp

    def initialize_belief(self, num_particles):
        pssnet = self.model_runner.model
        for view in self.robot_views:
            mean, logvar = pssnet.encode(stack_known(add_batch_to_dict(view.last_visible)))
            single_object_bel = ParticleBelief()
            single_object_bel.latent_prior_mean = mean
            single_object_bel.latent_prior_logvar = logvar
            single_object_bel.quantiles_log_pdf = compute_quantiles(mean, logvar, num_quantiles=100, num_samples=1000)

            for _ in range(num_particles):
                p = Particle()
                latent = pssnet.sample_latent_from_mean_and_logvar(mean, logvar)
                p.latent = tf.Variable(latent)
                p.sampled_latent = latent
                single_object_bel.particles.append(p)
            self.belief.particle_beliefs.append(single_object_bel)
        for _ in range(num_particles):
            self.belief.sampled_assignments.append([])

    def update_belief(self, obj_index: int, known_free, req, num_particles):
        self.reload_flow()
        bel = self.belief.particle_beliefs[obj_index]

        if len(bel.particles) > num_particles:
            raise RuntimeError("Unexpected situation - we have more particles than requested")

        if bel.latent_prior_mean is None:
            raise RuntimeError("Particle Belief was not initialized as expected")
            # self.initialize_belief(num_particles)

        if len(bel.particles) != num_particles:
            raise RuntimeError("Unexpected situation - number of particles does not match request")

        if self.method in ["proposed", "baseline_ignore_latent_prior", "VAE_GAN", "baseline_accept_failed_projections", "assign_all_CHS"]:
            self.update_belief_proposed(obj_index, known_free, req)
        elif self.method == "baseline_OOD_prediction":
            self.update_belief_OOD_prediction(obj_index, known_free, req)
        elif self.method == "baseline_rejection_sampling":
            self.update_belief_rejection_sampling(obj_index, known_free, req)
        elif self.method == "baseline_soft_rejection":
            self.update_belief_soft_rejection_sampling(obj_index, known_free, req)
        elif self.method == "baseline_direct_edit":
            self.update_belief_direct_edit(obj_index, known_free, req)
        else:
            raise RuntimeError(f"Unknown method {self.method}. Cannot update belief")

        if obj_index == 0:
            for i, p in enumerate(bel.particles):
                self.scene.goal_generator.publish_goal(p.goal, marker_id=i)

    def update_belief_proposed(self, obj_index, known_free, req):
        # First update current particles (If current particles exist, the prior mean and logvar must have been set
        for particle_num, particle in enumerate(self.belief.particle_beliefs[obj_index].particles):
            chss = self.get_assigned_chss(obj_index, req, particle_num)
            self.scene.goal_generator.clear_goal_markers()
            particle.latent, particle.successful_projection = self.enforce_contact(particle.latent, known_free, chss,
                                                                                   obj_index)
            predicted_occ = self.model_runner.model.decode(particle.latent, apply_sigmoid=True)
            pts = self.transform_to_gpuvoxels(self.robot_views[obj_index], predicted_occ)
            self.robot_views[obj_index].VG_PUB.publish('predicted_occ', predicted_occ)
            try:
                goal_tsr = self.scene.goal_generator.generate_goal_tsr(pts, publish=obj_index == 0)
            except RuntimeError as e:
                print(e)
                continue
            particle.goal = goal_tsr
            particle.completion = pts

    def update_belief_OOD_prediction(self, obj_index, known_free, req):
        visible_vg = deepcopy(self.robot_views[obj_index].last_visible)
        visible_vg['known_free'] += known_free

        for particle_num, particle in enumerate(self.belief.particle_beliefs[obj_index].particles):
            chss = self.get_assigned_chss(obj_index, req, particle_num)

            if chss is not None:
                gt = self.transform_from_gpuvoxels(self.robot_views[obj_index], self.scene.get_gt())
                gt = inflate_voxelgrid(tf.expand_dims(gt, axis=0))[0, :, :, :, :]
                contact_voxels = tf.reduce_sum(chss, axis=0)
                self.robot_views[obj_index].VG_PUB.publish('chs', contact_voxels)
                self.robot_views[obj_index].VG_PUB.publish('gt', gt)
                visible_vg['known_occ'] = np.clip(contact_voxels * gt + visible_vg['known_occ'], 0.0, 1.0)
                self.robot_views[obj_index].VG_PUB.publish('known_contact', contact_voxels * gt)

            self.scene.goal_generator.clear_goal_markers()
            self.robot_views[obj_index].VG_PUB.publish('known_free', visible_vg['known_free'])
            predicted_occ = self.model_runner.model.call(add_batch_to_dict(visible_vg), apply_sigmoid=True)[
                'predicted_occ']
            pts = self.transform_to_gpuvoxels(self.robot_views[obj_index], predicted_occ)
            self.robot_views[obj_index].VG_PUB.publish('predicted_occ', predicted_occ)
            try:
                goal_tsr = self.scene.goal_generator.generate_goal_tsr(pts)
            except RuntimeError as e:
                print(e)
                continue
            particle.goal = goal_tsr
            particle.completion = pts

    def update_belief_rejection_sampling(self, obj_index, known_free, req):
        for particle_num, particle in enumerate(self.belief.particle_beliefs[obj_index].particles):
            chss = self.get_assigned_chss(obj_index, req, particle_num)
            self.scene.goal_generator.clear_goal_markers()
            self.robot_views[obj_index].VG_PUB.publish('known_free', known_free)
            # predicted_occ = self.model_runner.model.call(add_batch_to_dict(self.robot_views[obj_index].last_visible),
            #                                              apply_sigmoid=True)['predicted_occ']
            predicted_occ = self.model_runner.model.decode(particle.latent, apply_sigmoid=True)

            pts = self.transform_to_gpuvoxels(self.robot_views[obj_index], predicted_occ)
            self.robot_views[obj_index].VG_PUB.publish('predicted_occ', predicted_occ)
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


    def update_belief_soft_rejection_sampling(self, obj_index, known_free, req):
        for particle_num, particle in enumerate(self.belief.particle_beliefs[obj_index].particles):
            chss = self.get_assigned_chss(obj_index, req, particle_num)
            self.scene.goal_generator.clear_goal_markers()
            self.robot_views[obj_index].VG_PUB.publish('known_free', known_free)
            # predicted_occ = self.model_runner.model.call(add_batch_to_dict(self.robot_views[obj_index].last_visible),
            #                                              apply_sigmoid=True)['predicted_occ']
            predicted_occ = self.model_runner.model.decode(particle.latent, apply_sigmoid=True)

            pts = self.transform_to_gpuvoxels(self.robot_views[obj_index], predicted_occ)
            self.robot_views[obj_index].VG_PUB.publish('predicted_occ', predicted_occ)
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
                particle.constraint_violation_count += tf.reduce_sum(known_free * predicted_occ)
            if not (predicted_occ, chss):
                particle.successful_projection = False
                unsatisfied_chs_count = tf.reduce_max(chss * predicted_occ, axis=(1, 2, 3, 4)) #TODO This is what I was last working on
                particle.constraint_violation_count += unsatisfied_chs_count

    def update_belief_direct_edit(self, obj_index, known_free, req):
        # First update current particles (If current particles exist, the prior mean and logvar must have been set
        for particle_num, particle in enumerate(self.belief.particle_beliefs[obj_index].particles):
            chss = self.get_assigned_chss(obj_index, req, particle_num)
            self.scene.goal_generator.clear_goal_markers()
            self.robot_views[obj_index].VG_PUB.publish('known_free', known_free)
            # particle.latent, particle.successful_projection = self.enforce_contact(particle.latent, known_free, chss,
            #                                                                        obj_index)
            # predicted_occ = self.model_runner.model.decode(particle.latent, apply_sigmoid=True)
            predicted_occ = self.model_runner.model.decode(particle.latent, apply_sigmoid=True)

            # Directly remove any known free
            predicted_occ = np.clip(predicted_occ - inflate_voxelgrid(known_free, is_batched=False), 0, 1)

            # Directly add any voxels in contact
            if chss is not None:
                gt = self.transform_from_gpuvoxels(self.robot_views[obj_index], self.scene.get_gt())
                gt = inflate_voxelgrid(tf.expand_dims(gt, axis=0))[0, :, :, :, :]
                contact_voxels = tf.reduce_sum(chss, axis=0)
                self.robot_views[obj_index].VG_PUB.publish('chs', contact_voxels)
                self.robot_views[obj_index].VG_PUB.publish('gt', gt)
                predicted_occ = np.clip(contact_voxels * gt + predicted_occ, 0.0, 1.0)
                self.robot_views[obj_index].VG_PUB.publish('known_contact', contact_voxels * gt)

            pts = self.transform_to_gpuvoxels(self.robot_views[obj_index], predicted_occ)
            self.robot_views[obj_index].VG_PUB.publish('predicted_occ', predicted_occ)

            try:
                goal_tsr = self.scene.goal_generator.generate_goal_tsr(pts, publish=obj_index == 0)
            except RuntimeError as e:
                print(e)
                continue
            particle.goal = goal_tsr
            particle.completion = pts

    @staticmethod
    def transform_from_gpuvoxels(view, pt_msg: PointCloud2):
        transformed_cloud = view.transform_pts_to_target(pt_msg)
        xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(transformed_cloud)
        # TODO: visual_conversions produces the wrong result cause the transforms are calculated differently.
        #  Look into this
        vg = conversions.pointcloud_to_voxelgrid(xyz_array, scale=view.scale,
                                                 origin=view.origin,
                                                 add_trailing_dim=True, add_leading_dim=False,
                                                 )
        return vg

    def transform_to_gpuvoxels(self, view, vg) -> PointCloud2:

        # TODO: It is odd that I use visual_conversions here, since I used conversions (not visual, different package
        #  of mine) get the pointcloud in the first place. However, visual_conversions has this nice function which
        #  densifies the points
        msg = visual_conversions.vox_to_pointcloud2_msg(vg, frame=view.target_frame,
                                                        scale=view.scale,
                                                        origin=-view.origin / view.scale,
                                                        density_factor=self.completion_density)
        # pt = conversions.voxelgrid_to_pointcloud(vg, scale=self.robot_view.scale,
        #                                          origin=self.robot_view.origin)

        msg = view.transform_pts_to_target(msg, target_frame="gpu_voxel_world")
        return msg

    def enforce_contact(self, latent, known_free, chss, obj_index, verbose=True):
        if self.method in ["proposed", "baseline_accept_failed_projections", "VAE_GAN", "assign_all_CHS"]:
            return enforce_contact(latent, known_free, chss, self.model_runner.model,
                                   self.belief.particle_beliefs[obj_index], self.robot_views[obj_index].VG_PUB, verbose)
        elif self.method == "baseline_ignore_latent_prior":
            return enforce_contact_ignore_latent_prior(latent, known_free, chss, self.model_runner.model,
                                                       self.belief.particle_beliefs[obj_index],
                                                       self.robot_views[obj_index].VG_PUB)
        else:
            raise RuntimeError(f"Not known how to enforce_contact for method {self.method}")
