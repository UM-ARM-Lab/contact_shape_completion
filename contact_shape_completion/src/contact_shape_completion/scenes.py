import abc
import argparse
from pathlib import Path
from typing import List

import numpy as np
import rospkg
import rospy
from colorama import Fore
from sensor_msgs.msg import PointCloud2

from enum import Enum

import ros_numpy
from arc_utilities import ros_helpers
from contact_shape_completion.goal_generator import GoalGenerator, CheezeitGoalGenerator, PitcherGoalGenerator, \
    LiveCheezitGoalGenerator, LivePitcherGoalGenerator
from rviz_voxelgrid_visuals import conversions as visual_conversions
from rviz_voxelgrid_visuals.conversions import get_origin_in_voxel_coordinates, points_to_pointcloud2_msg
from shape_completion_training.model import default_params
from shape_completion_training.utils import dataset_loader
from shape_completion_training.utils.data_tools import shift_voxelgrid


def get_scene(scene_name: str):
    scene_map = {"live_cheezit": LiveScene1,
                 "cheezit_01": SimulationCheezit,
                 "cheezit_deep": SimulationDeepCheezit,
                 "pitcher": SimulationPitcher,
                 "mug": SimulationMug,
                 "live_pitcher": LivePitcher,
                 "multiobject": SimulationMultiObject,
                 }
    if scene_name not in scene_map:
        print(f"{Fore.RED}Unknown scene name {scene_name}\nValid scene names are:")
        print(f"{list(scene_map.keys())}{Fore.RESET}")
        raise RuntimeError(f"Unknown scene name {scene_name}")
    return scene_map[scene_name]()


def combine_pointcloud2s(pt_clouds: List[PointCloud2]):
    if len(pt_clouds) == 1:
        return pt_clouds[0]

    xyz_array = np.zeros((0, 3))
    frame = pt_clouds[0].header.frame_id
    for pts in pt_clouds:
        if pts.header.frame_id != frame:
            raise RuntimeError(f"Cannot combine pointclouds with frames {frame} and {pts.header.frame_id}")
        xyz_array = np.concatenate([xyz_array, ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pts)], axis=0)
    return points_to_pointcloud2_msg(xyz_array, frame=frame)


class SceneType(Enum):
    LIVE = 1
    SIMULATION = 2
    UNSPECIFIED = 3


class Scene(abc.ABC):
    scene_type = SceneType.UNSPECIFIED
    num_objects = 1

    def __init__(self):
        self.goal_generator = None
        self.scale = None
        self.name = None
        self.goal_generator: GoalGenerator
        self.forward_shift_for_voxelgrid = 0.1
        self.segmented_object_categories = [1, 2]  # Cheezit

    @abc.abstractmethod
    def get_gt(self):
        pass

    def get_segmented_points(self):
        pass

    def get_save_path(self):
        path = Path(rospkg.RosPack().get_path("contact_shape_completion")) / "saved_requests" / self.name
        path.parent.mkdir(exist_ok=True)
        path.mkdir(exist_ok=True)
        return path


class SimulationScene(Scene):
    scene_type = SceneType.SIMULATION


class SimulationCheezit(SimulationScene):
    def __init__(self):
        super().__init__()
        self.name = "Cheezit_01"
        self.dataset_supervisor = dataset_loader.get_dataset_supervisor('ycb_all')
        params = default_params.get_noiseless_params()
        params['apply_depth_sensor_noise'] = True
        self.elem = self.dataset_supervisor.get_element('003_cracker_box-90_000_000',
                                                        params=params).load()
        self.scale = 0.01
        self.origin = get_origin_in_voxel_coordinates((1.2, 1.6, 1.2), self.scale)
        self.goal_generator = CheezeitGoalGenerator()

    def get_gt(self, density_factor=3):
        pts = visual_conversions.vox_to_pointcloud2_msg(self.elem['gt_occ'], scale=self.scale, frame='gpu_voxel_world',
                                                        origin=self.origin,
                                                        density_factor=density_factor)
        return pts

    def get_segmented_points(self):
        pts = visual_conversions.vox_to_pointcloud2_msg(self.elem['known_occ'], scale=self.scale,
                                                        frame='gpu_voxel_world',
                                                        origin=self.origin,
                                                        density_factor=3)
        return [pts]


class SimulationDeepCheezit(SimulationScene):
    def __init__(self):
        super().__init__()
        self.name = "Cheezit_deep"
        self.dataset_supervisor = dataset_loader.get_dataset_supervisor('ycb_all')
        params = default_params.get_noiseless_params()
        params['apply_depth_sensor_noise'] = True
        self.elem = self.dataset_supervisor.get_element('003_cracker_box-90_000_000',
                                                        params=params).load()
        gt = self.elem['gt_occ'][0, :, :, :, :]
        for _ in range(3):
            gt = shift_voxelgrid(gt, 4, 0, 0, pad_value=0, max_x=4, max_y=0, max_z=0)
            self.elem['gt_occ'] = np.clip(self.elem['gt_occ'] + gt, 0.0, 1.0)
        self.scale = 0.01
        self.origin = get_origin_in_voxel_coordinates((1.0, 1.6, 1.2), self.scale)
        self.goal_generator = CheezeitGoalGenerator()
        self.forward_shift_for_voxelgrid = 0.2

    def get_gt(self, density_factor=3):
        pts = visual_conversions.vox_to_pointcloud2_msg(self.elem['gt_occ'], scale=self.scale, frame='gpu_voxel_world',
                                                        origin=self.origin,
                                                        density_factor=density_factor)
        return pts

    def get_segmented_points(self):
        pts = visual_conversions.vox_to_pointcloud2_msg(self.elem['known_occ'], scale=self.scale,
                                                        frame='gpu_voxel_world',
                                                        origin=self.origin,
                                                        density_factor=3)
        return [pts]


class SimulationPitcher(SimulationScene):
    def __init__(self):
        super().__init__()
        self.name = "pitcher"
        self.dataset_supervisor = dataset_loader.get_dataset_supervisor('ycb_all')
        params = default_params.get_noiseless_params()
        params['apply_depth_sensor_noise'] = True
        self.elem = self.dataset_supervisor.get_element('019_pitcher_base-90_000_000',
                                                        params=params).load()
        self.scale = 0.007
        self.origin = get_origin_in_voxel_coordinates((1.2, 2.0, 1.2), self.scale)
        self.goal_generator = PitcherGoalGenerator(x_bound=(-0.01, 0.01))

    def get_gt(self, density_factor=3):
        pts = visual_conversions.vox_to_pointcloud2_msg(self.elem['gt_occ'], scale=self.scale, frame='gpu_voxel_world',
                                                        origin=self.origin,
                                                        density_factor=density_factor)
        return pts

    def get_segmented_points(self):
        pts = visual_conversions.vox_to_pointcloud2_msg(self.elem['known_occ'], scale=self.scale,
                                                        frame='gpu_voxel_world',
                                                        origin=self.origin,
                                                        density_factor=3)
        return [pts]


class SimulationMug(SimulationScene):
    def __init__(self):
        super().__init__()
        self.name = "mug"
        self.dataset_supervisor = dataset_loader.get_dataset_supervisor('shapenet_mugs')
        params = default_params.get_noiseless_params()
        params['apply_depth_sensor_noise'] = True
        self.elem = self.dataset_supervisor.get_element('10c2b3eac377b9084b3c42e318f3affc000_260_000',
                                                        params=params).load()
        self.scale = 0.007
        self.origin = get_origin_in_voxel_coordinates((1.2, 2.0, 1.2), self.scale)
        self.goal_generator = PitcherGoalGenerator(x_bound=(-0.01, 0.01))

    def get_gt(self, density_factor=3):
        pts = visual_conversions.vox_to_pointcloud2_msg(self.elem['gt_occ'], scale=self.scale, frame='gpu_voxel_world',
                                                        origin=self.origin,
                                                        density_factor=density_factor)
        return pts

    def get_segmented_points(self):
        pts = visual_conversions.vox_to_pointcloud2_msg(self.elem['known_occ'], scale=self.scale,
                                                        frame='gpu_voxel_world',
                                                        origin=self.origin,
                                                        density_factor=3)
        return [pts]


class SimulationMultiObject(SimulationScene):
    num_objects = 2

    def __init__(self):
        super().__init__()

        self.name = "multiobject"
        self.dataset_supervisor = dataset_loader.get_dataset_supervisor('ycb_all')
        params = default_params.get_noiseless_params()
        params['apply_depth_sensor_noise'] = True
        self.elems = [self.dataset_supervisor.get_element('019_pitcher_base-90_000_000',
                                                          params=params).load(),
                      self.dataset_supervisor.get_element('003_cracker_box-90_000_000',
                                                          params=params).load()
                      ]

        self.scale = 0.007
        self.origins = [get_origin_in_voxel_coordinates((1.2, 2.0, 1.2), self.scale),
                        get_origin_in_voxel_coordinates((1.2, 1.6, 1.2), self.scale)]

        self.goal_generator = PitcherGoalGenerator(x_bound=(-0.01, 0.01))

    def get_gt(self, density_factor=3):
        def conv(elem, origin):
            return visual_conversions.vox_to_pointcloud2_msg(elem['gt_occ'], scale=self.scale, frame='gpu_voxel_world',
                                                             origin=origin,
                                                             density_factor=density_factor)
        pts_all_objects = [conv(elem, origin) for elem, origin in zip(self.elems, self.origins)]
        return combine_pointcloud2s(pts_all_objects)

    def get_segmented_points(self):
        def conv(elem, origin):
            return visual_conversions.vox_to_pointcloud2_msg(elem['known_occ'], scale=self.scale,
                                                             frame='gpu_voxel_world',
                                                             origin=origin,
                                                             density_factor=3)

        pts = [conv(elem, origin) for elem, origin in zip(self.elems, self.origins)]
        return pts


class LiveScene(Scene):
    scene_type = SceneType.LIVE

    def __init__(self):
        super().__init__()
        self.goal_generator = LiveCheezitGoalGenerator(x_bound=(-0.01, 0.01))


class LiveScene1(LiveScene):
    def __init__(self):
        super().__init__()
        self.name = "live_cheezit"

    def get_gt(self, density_factor=3):
        vg = np.ones((8, 8, 11))
        scale = 0.02
        origin = visual_conversions.get_origin_in_voxel_coordinates((1.45, 1.92, 1.43), scale=scale)
        pts = visual_conversions.vox_to_pointcloud2_msg(vg, scale=scale, frame='gpu_voxel_world',
                                                        origin=origin,
                                                        density_factor=3)
        return pts


class LivePitcher(LiveScene):
    def __init__(self):
        super().__init__()
        self.name = "live_pitcher"
        self.dataset_supervisor = dataset_loader.get_dataset_supervisor('ycb_all')
        params = default_params.get_noiseless_params()
        params['apply_depth_sensor_noise'] = True
        self.elem = self.dataset_supervisor.get_element('019_pitcher_base-90_000_330',
                                                        params=params).load()
        self.scale = 0.0065
        self.origin = get_origin_in_voxel_coordinates((1.3, 1.67, 1.33), self.scale)
        self.goal_generator = LivePitcherGoalGenerator(x_bound=(-0.01, 0.01))
        self.segmented_object_categories = [11]

    def get_gt(self, density_factor=3):
        pts = visual_conversions.vox_to_pointcloud2_msg(self.elem['gt_occ'], scale=self.scale, frame='gpu_voxel_world',
                                                        origin=self.origin,
                                                        density_factor=density_factor)
        return pts


def visualize(pts, point_pub: rospy.Publisher):
    point_pub.publish(pts)


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
    # parser.add_argument('--trial')
    parser.add_argument('--scene')
    return parser.parse_args()


def main():
    rospy.init_node("simulation_scene")
    point_pub = ros_helpers.get_connected_publisher('/pointcloud', PointCloud2, queue_size=1)
    args = parse_command_line_args()

    if args.scene is None:
        print(f"{Fore.RED}Must provide a scene name.{Fore.RESET}")
        exit()

    pts = get_scene(args.scene).get_gt()

    visualize(pts, point_pub)
    rospy.sleep(1)


if __name__ == "__main__":
    main()
