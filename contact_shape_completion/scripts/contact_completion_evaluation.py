#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
import rospkg
import rospy
import seaborn as sns
from colorama import Fore
from matplotlib import pyplot as plt
from sensor_msgs.msg import PointCloud2

from arc_utilities import ros_helpers
from contact_shape_completion import scenes
from contact_shape_completion.contact_shape_completer import ContactShapeCompleter
from contact_shape_completion.evaluation import pt_cloud_distance
from contact_shape_completion.goal_generator import CheezeitGoalGenerator
from gpu_voxel_planning_msgs.srv import CompleteShapeRequest
from shape_completion_training.model import default_params
from shape_completion_training.utils.config import lookup_trial
from dataclasses import dataclass


default_dataset_params = default_params.get_default_params()


@dataclass()
class EvaluationDetails:
    scene_type: Type[scenes.Scene]
    network: str
    method: str


def get_evaluation_trials():
    d = [EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='proposed')]
    return d


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
    # parser.add_argument('--trial')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()


def generate_evaluation(details):
    # x_bound = (-0.004, 0.004)
    x_bound = [-0.04, 0.04]
    scene = details.scene_type()

    goal_generator = CheezeitGoalGenerator(x_bound=x_bound)
    # scene = simulation_ground_truth_scenes.LiveScene1()

    contact_shape_completer = ContactShapeCompleter(scene, lookup_trial(details.network),
                                                    goal_generator=goal_generator,
                                                    completion_density=1)

    contact_shape_completer.get_visible_vg()

    columns = ['request number', 'chs count', 'particle num', 'scene', 'method', 'chamfer distance',
               'projection succeeded?']
    df = pd.DataFrame(columns=columns)

    gt = scene.get_gt(density_factor=1)
    point_pub = ros_helpers.get_connected_publisher('/pointcloud', PointCloud2, queue_size=1)
    point_pub.publish(gt)

    files = sorted([f for f in scene.get_save_path().glob('*')])
    for req_number, file in enumerate(files):
        print(f"{Fore.CYAN}Loading stored request{file}{Fore.RESET}")
        completion_req = CompleteShapeRequest()
        with file.open('rb') as f:
            completion_req.deserialize(f.read())

        completion_req.num_samples = 10
        resp = contact_shape_completer.complete_shape_srv(completion_req)
        dists = []
        for particle_num, completion_pts_msg in enumerate(resp.sampled_completions):
            dist = pt_cloud_distance(completion_pts_msg, gt).numpy()
            print(f"Errors w.r.t. gt: {dist}")
            dists.append(dist)
            df = df.append(pd.Series([
                req_number, float(len(completion_req.chss)), particle_num, scene.name, 'wip_method', dist, True
            ], index=columns), ignore_index=True)
        print(f"Closest particle has error {np.min(dists)}")
        print(f"Average particle has error {np.mean(dists)}")
        # display_sorted_particles(resp.sampled_completions, dists)

    df.to_csv(get_evaluation_path(details))
    print(df)


def display_sorted_particles(particles, dists):
    ordered_particles = sorted(zip(particles, dists), key=lambda x: x[1])
    point_pub = ros_helpers.get_connected_publisher('/pointcloud', PointCloud2, queue_size=1)
    for particle, dist in ordered_particles:
        point_pub.publish(particle)
        print(f"Dist {dist}")
        rospy.sleep(0.1)


def get_evaluation_path(details: EvaluationDetails):
    scene = details.scene_type()
    path = Path(rospkg.RosPack().get_path("contact_shape_completion")) / "evaluations" / scene.name / \
           f'{details.network}_{details.method}.csv'
    path.parent.parent.mkdir(exist_ok=True)
    path.parent.mkdir(exist_ok=True)
    return path


def plot(details: EvaluationDetails):
    scene = details.scene_type()
    df = pd.read_csv(get_evaluation_path(details))
    print(f"Plotting {scene.name}, {details.network}, {details.method}")
    df = df[['request number', 'chs count', 'chamfer distance']] \
        .groupby('request number', as_index=False) \
        .agg({'request number': 'first', 'chamfer distance': ['mean', 'min', 'max']})
    err = df[[('chamfer distance', 'min'), ('chamfer distance', 'max')]]
    plt.rcParams['errorbar.capsize'] = 10
    sns.barplot(x=('request number', 'first'), y=('chamfer distance', 'mean'), data=df, yerr=err.T.to_numpy())
    plt.show()


def main():
    ARGS = parse_command_line_args()

    rospy.init_node('contact_completion_evaluation')
    rospy.loginfo("Contact Completion Evaluation")

    # scene = scenes.SimulationCheezit()
    for details in get_evaluation_trials():
        if ARGS.regenerate:
            print(f'{Fore.CYAN}Regenerating {details}{Fore.RESET}')
            generate_evaluation(details)
        elif not get_evaluation_path(details).exists():
            print(f'{Fore.CYAN}Generating {details}{Fore.RESET}')
            generate_evaluation(details)
        else:
            print(f"{details} exists. Not generating")

        if ARGS.plot:
            plot(details)


if __name__ == "__main__":
    main()
