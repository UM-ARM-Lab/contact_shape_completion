import abc
import argparse
from pathlib import Path

import numpy as np
import rospkg
import rospy
from colorama import Fore
from sensor_msgs.msg import PointCloud2

from arc_utilities import ros_helpers
from rviz_voxelgrid_visuals import conversions as visual_conversions
from rviz_voxelgrid_visuals.conversions import get_origin_in_voxel_coordinates
from shape_completion_training.model import default_params
from shape_completion_training.utils import dataset_loader
from shape_completion_training.utils.data_tools import shift_voxelgrid


def get_scene(scene_name: str):
    scene_map = {"live_01": LiveScene1,
                 "cheezit_01": SimulationCheezit,
                 "cheezit_deep": SimulationDeepCheezit}
    if scene_name not in scene_map:
        print(f"{Fore.RED}Unknown scene name {scene_name}\nValid scene names are:")
        print(f"{list(scene_map.keys())}{Fore.RESET}")
        raise RuntimeError(f"Unknown scene name {scene_name}")
    return scene_map[scene_name]()


class Scene(abc.ABC):
    def __init__(self):
        self.use_live = False
        self.name = None

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
    pass


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
        return pts


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
        return pts


class LiveScene(Scene):
    def __init__(self):
        super().__init__()
        self.use_live = True


class LiveScene1(LiveScene):
    def get_gt(self):
        vg = np.ones((5, 9, 17))
        pts = visual_conversions.vox_to_pointcloud2_msg(vg, scale=0.02, frame='gpu_voxel_world',
                                                        origin=(-72, -100, -65),
                                                        density_factor=3)
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
