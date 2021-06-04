#!/usr/bin/env python
import argparse

import numpy as np
import rospy
from colorama import Fore
from sensor_msgs.msg import PointCloud2

from arc_utilities import ros_helpers
from contact_shape_completion import scenes
from contact_shape_completion.contact_shape_completer import ContactShapeCompleter
from contact_shape_completion.evaluation import pt_cloud_distance
from contact_shape_completion.goal_generator import CheezeitGoalGenerator
from gpu_voxel_planning_msgs.srv import CompleteShapeRequest
from shape_completion_training.model import default_params
from shape_completion_training.utils.config import lookup_trial
import pandas as pd

"""
Publish object pointclouds for use in gpu_voxels planning
"""

default_dataset_params = default_params.get_default_params()


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
    parser.add_argument('--trial')
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_command_line_args()

    rospy.init_node('contact_shape_completer_service')
    rospy.loginfo("Data Publisher")

    # x_bound = (-0.004, 0.004)
    x_bound = [-0.04, 0.04]

    goal_generator = CheezeitGoalGenerator(x_bound=x_bound)
    # scene = simulation_ground_truth_scenes.LiveScene1()
    scene = scenes.SimulationCheezit()
    contact_shape_completer = ContactShapeCompleter(scene, lookup_trial(ARGS.trial), goal_generator=goal_generator,
                                                    completion_density=1)
    # contact_shape_completer.load_network(ARGS.trial)

    # contact_shape_completer.load_visible_vg(filename='wip_segmented_pts.msg')
    contact_shape_completer.get_visible_vg()

    df = pd.DataFrame(columns=["trial", ""])

    gt = scene.get_gt(density_factor=1)
    point_pub = ros_helpers.get_connected_publisher('/pointcloud', PointCloud2, queue_size=1)
    point_pub.publish(gt)

    files = sorted([f for f in scene.get_save_path().glob('*')])
    for file in files:
        print(f"{Fore.CYAN}Loading stored request{file}{Fore.RESET}")
        completion_req = CompleteShapeRequest()
        with file.open('rb') as f:
            completion_req.deserialize(f.read())

        resp = contact_shape_completer.complete_shape_srv(completion_req)
        dists = []
        for completion_pts_msg in resp.sampled_completions:
            dist = pt_cloud_distance(completion_pts_msg, gt)
            print(f"Errors w.r.t. gt: {dist}")
            dists.append(dist)
        print(f"Closest particle has error {np.min(dists)}")
        print(f"Average particle has error {np.mean(dists)}")


