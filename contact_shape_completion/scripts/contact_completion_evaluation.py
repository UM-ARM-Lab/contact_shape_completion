#!/usr/bin/env python
import argparse
from pathlib import Path

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
from contact_shape_completion.evaluation import pt_cloud_distance, vg_chamfer_distance
from contact_shape_completion.goal_generator import CheezeitGoalGenerator

from contact_shape_completion.evaluation_params import EvaluationDetails
from gpu_voxel_planning_msgs.srv import CompleteShapeRequest
from shape_completion_training.model import default_params
from shape_completion_training.utils.config import lookup_trial

default_dataset_params = default_params.get_default_params()

NUM_PARTICLES_IN_TRIAL = 100


def get_evaluation_trials():
    d = [EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='proposed'),
         EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='baseline_ignore_latent_prior'),
         EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='baseline_OOD_prediction'),
         EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='baseline_rejection_sampling'),
         EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB', method='proposed'),
         EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB', method='baseline_ignore_latent_prior'),
         EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB', method='baseline_OOD_prediction'),
         EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB', method='baseline_rejection_sampling'),
         ]
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
                                                    completion_density=1,
                                                    method=details.method)

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

        completion_req.num_samples = NUM_PARTICLES_IN_TRIAL
        resp = contact_shape_completer.complete_shape_srv(completion_req)
        dists = []
        for particle_num, completion_pts_msg in enumerate(resp.sampled_completions):
            # dist = pt_cloud_distance(completion_pts_msg, gt).numpy()
            dist = vg_chamfer_distance(contact_shape_completer.transform_from_gpuvoxels(completion_pts_msg),
                                       contact_shape_completer.transform_from_gpuvoxels(gt),
                                       scale=contact_shape_completer.robot_view.scale).numpy()
            print(f"Errors w.r.t. gt: {dist}")
            dists.append(dist)
            df = df.append(pd.Series([
                req_number, float(len(completion_req.chss)), particle_num, scene.name, 'wip_method', dist, True
            ], index=columns), ignore_index=True)
        if len(dists) == 0:
            continue
        print(f"Closest particle has error {np.min(dists)}")
        print(f"Average particle has error {np.mean(dists)}")
        display_sorted_particles(resp.sampled_completions, dists)

    df.to_csv(get_evaluation_path(details))
    print(df)
    contact_shape_completer.unsubscribe()


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


def percentile_fun(n):
    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def plot(details: EvaluationDetails):
    scene = details.scene_type()
    df = pd.read_csv(get_evaluation_path(details))

    bar_key = ('chamfer distance', 'median')
    error_key_lower = ('chamfer distance', 'percentile_25')
    # error_key_lower = ('chamfer distance', 'min')
    error_key_upper = ('chamfer distance', 'percentile_75')

    print(f"Plotting {scene.name}, {details.network}, {details.method}")
    df = df[['request number', 'chs count', 'chamfer distance']] \
        .groupby('request number', as_index=False) \
        .agg({'request number': 'first',
              'chamfer distance': ['mean', 'min', 'max', 'median', percentile_fun(25), percentile_fun(75)]})
    # err = df[[('chamfer distance', 'min'), ('chamfer distance', 'max')]]
    df['bar min'] = df[bar_key] - df[error_key_lower]
    df['bar max'] = df[error_key_upper] - df[bar_key]
    err = df[['bar min', 'bar max']]

    plt.rcParams['errorbar.capsize'] = 10
    ax = sns.barplot(x=('request number', 'first'), y=bar_key, data=df, yerr=err.T.to_numpy())
    ax.set_xlabel('Number of observations')
    ax.set_ylabel('Chamfer Distance to True Scene')
    ax.set_title(f'{scene.name}: {details.method}')
    plt.savefig(f'/home/bsaund/Pictures/shape contact/{scene.name}_{details.method}')
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
