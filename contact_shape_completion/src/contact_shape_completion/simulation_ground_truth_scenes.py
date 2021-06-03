import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2

from arc_utilities import ros_helpers
from rviz_voxelgrid_visuals import conversions as visual_conversions
from rviz_voxelgrid_visuals.conversions import get_origin_in_voxel_coordinates
from shape_completion_training.model import default_params
from shape_completion_training.utils import dataset_loader
import abc


class Scene(abc.ABC):
    def __init__(self):
        self.use_live = False

    @abc.abstractmethod
    def get_gt(self):
        pass

    def get_segmented_points(self):
        pass


class SimulationScene(Scene):
    pass


class SimulationCheezit(SimulationScene):
    def __init__(self):
        super().__init__()
        self.dataset_supervisor = dataset_loader.get_dataset_supervisor('ycb_all')
        params = default_params.get_noiseless_params()
        params['apply_depth_sensor_noise']= True
        self.elem = self.dataset_supervisor.get_element('003_cracker_box-90_000_000',
                                                        params=params).load()
        self.scale = 0.01
        self.origin = get_origin_in_voxel_coordinates((1.2, 1.6, 1.2), self.scale)

    def get_gt(self):
        pts = visual_conversions.vox_to_pointcloud2_msg(self.elem['gt_occ'], scale=self.scale, frame='gpu_voxel_world',
                                                        origin=self.origin,
                                                        density_factor=3)
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


def main():
    rospy.init_node("simulation_scene")
    point_pub = ros_helpers.get_connected_publisher('/pointcloud', PointCloud2, queue_size=1)
    pts = LiveScene1().get_gt()

    visualize(pts, point_pub)
    rospy.sleep(1)


if __name__ == "__main__":
    main()
