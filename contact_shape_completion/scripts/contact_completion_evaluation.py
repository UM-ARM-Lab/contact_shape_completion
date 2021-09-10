#!/usr/bin/env python
import argparse
import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import rospkg
import rospy
import seaborn as sns
from colorama import Fore
from matplotlib import pyplot as plt
from sensor_msgs.msg import PointCloud2

import tensorflow as tf

from arc_utilities import ros_helpers
from contact_shape_completion import scenes
from contact_shape_completion.contact_shape_completer import ContactShapeCompleter
from contact_shape_completion.evaluation import vg_chamfer_distance
from contact_shape_completion.evaluation_params import EvaluationDetails
from gpu_voxel_planning_msgs.srv import CompleteShapeRequest
from shape_completion_training.model import default_params
from shape_completion_training.utils.config import lookup_trial
import os
import pwd

YOUR_USERNAME = pwd.getpwuid(os.getuid())[0]
root_save_path = f'/home/{YOUR_USERNAME}/Pictures/shape contact'

default_dataset_params = default_params.get_default_params()

NUM_PARTICLES_IN_TRIAL = 100

display_names_map = {
    'live_cheezit': 'Live Cheezit',
    'Cheezit_01': 'Simulation Cheezit (Shallow)',
    'Cheezit_deep': 'Simulation Cheezit (Deep)',
    'pitcher': 'Simulation Pitcher',
    'mug': 'Simulation Mug',
    'live_pitcher': 'Live Pitcher',
    'multiobject': 'MultiObject',
    'live_multiobject': 'Live MultiObject',
    'proposed': 'PSSNet + CLASP',
    'baseline_ignore_latent_prior': 'PSSNet + CLASP: ignore prior',
    'baseline_accept_failed_projections': 'PSSNet + CLASP: accept failed projections',
    'baseline_OOD_prediction': 'PSSNet OOD',
    'baseline_rejection_sampling': "PSSNet Rejection Sampling",
    'VAE_GAN': "VAE_GAN + CLASP",
    'assign_all_CHS': "PSSNet + Clasp: No contact disambiguation",
    'baseline_direct_edit': "Direct Edit",
    'baseline_soft_rejection': "PSSNet Soft Rejection Sampling",
    'proposed_with_skin': "PSSNet + CLASP with skin"
}

display_legends_for = [
    'Simulation Cheezit (Shallow)',
    'MultiObject'
]

observations_not_displayed = {
    'Simulation Pitcher': [5, 7, 8, 9],
    'Simulation Cheezit (Shallow)': [1, 7],
    'Live Cheezit': [7, 8, 9]
}

linesize = defaultdict(lambda: 1.0)
linesize['PSSNet + CLASP'] = 2.5


@dataclasses.dataclass(frozen=True)
class PlottingParams:
    name: str
    show_legend: bool


def get_evaluation_trial_groups():
    d = {
        # [
        #     EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='proposed'),
        #     EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB',
        #                       method='baseline_ignore_latent_prior'),
        #     EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB',
        #                       method='baseline_accept_failed_projections'),
        #     EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='baseline_OOD_prediction'),
        #     EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='baseline_rejection_sampling'),
        #     EvaluationDetails(scene_type=scenes.SimulationCheezit, network='VAE_GAN_aab',
        #                       method='VAE_GAN'),
        # ],
        # [
        #     EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB', method='proposed'),
        #     EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB',
        #                       method='baseline_ignore_latent_prior'),
        #     EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB',
        #                       method='baseline_accept_failed_projections'),
        #     EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB', method='baseline_OOD_prediction'),
        #     EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB',
        #                       method='baseline_rejection_sampling'),
        #     EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='VAE_GAN_aab',
        #                       method='VAE_GAN'),
        # ],
        # [
        #     EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB', method='proposed'),
        #     EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB',
        #                       method='baseline_ignore_latent_prior'),
        #     EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB',
        #                       method='baseline_accept_failed_projections'),
        #     EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB', method='baseline_OOD_prediction'),
        #     EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB',
        #                       method='baseline_rejection_sampling'),
        #     EvaluationDetails(scene_type=scenes.SimulationPitcher, network='VAE_GAN_YCB',
        #                       method='VAE_GAN'),
        # ],
        # [
        #     # EvaluationDetails(scene_type=scenes.LiveScene1, network='AAB',
        #     #                   method='proposed'),
        #     EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB',
        #                       method='proposed'),
        #     EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB',
        #                       method='baseline_ignore_latent_prior'),
        #     EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB',
        #                       method='baseline_accept_failed_projections'),
        #     EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB',
        #                       method='baseline_OOD_prediction'),
        #     EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB',
        #                       method='baseline_rejection_sampling'),
        #     EvaluationDetails(scene_type=scenes.LiveScene1, network='VAE_GAN_YCB',
        #                       method='VAE_GAN'),
        # ],
        # [
        #     EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs',
        #                       method='proposed'),
        #     EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs',
        #                       method='baseline_ignore_latent_prior'),
        #     EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs',
        #                       method='baseline_accept_failed_projections'),
        #     EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs',
        #                       method='baseline_OOD_prediction'),
        #     EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs',
        #                       method='baseline_rejection_sampling'),
        #     EvaluationDetails(scene_type=scenes.SimulationMug, network='VAE_GAN_mugs',
        #                       method='VAE_GAN'),
        # ],
        # [
        #     EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB',
        #                       method='proposed'),
        #     EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB',
        #                       method='baseline_ignore_latent_prior'),
        #     EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB',
        #                       method='baseline_accept_failed_projections'),
        #     EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB',
        #                       method='baseline_OOD_prediction'),
        #     EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB',
        #                       method='baseline_rejection_sampling'),
        #     EvaluationDetails(scene_type=scenes.LivePitcher, network='VAE_GAN_YCB',
        #                       method='VAE_GAN'),
        # ],
        # [
        #     EvaluationDetails(scene_type=scenes.SimulationMultiObject, network='YCB',
        #                       method='proposed'),
        #     EvaluationDetails(scene_type=scenes.SimulationMultiObject, network='YCB',
        #                       method='assign_all_CHS'),
        #     EvaluationDetails(scene_type=scenes.SimulationMultiObject, network='VAE_GAN_YCB',
        #                       method='VAE_GAN'),
        # ],

        # [
        #     EvaluationDetails(scene_type=scenes.LiveMultiObject, network='YCB',
        #                       method='proposed'),
        # EvaluationDetails(scene_type=scenes.SimulationMultiObject, network='YCB',
        #                   method='assign_all_CHS'),
        # EvaluationDetails(scene_type=scenes.SimulationMultiObject, network='VAE_GAN_YCB',
        #                   method='VAE_GAN'),
        # ]



        ######################################
        ### CoRL Revisions: More Baselines ###
        ######################################

        PlottingParams('Baselines for Simulation Cheezit (Shallow): AAB', True):
            [
                EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='proposed'),
                EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB',
                                  method='baseline_direct_edit'),
                EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB',
                                  method='baseline_rejection_sampling'),
                EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='baseline_soft_rejection'),
                EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='baseline_OOD_prediction'),
                EvaluationDetails(scene_type=scenes.SimulationCheezit, network='VAE_GAN_aab',
                                  method='VAE_GAN'),
            ],
        PlottingParams('Ablations for Simulation Cheezit (Shallow): AAB', True):
            [
                EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='proposed'),
                    EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB',
                                      method='baseline_ignore_latent_prior'),
                    EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB',
                                      method='baseline_accept_failed_projections'),
            ],
        PlottingParams('Baselines for Simulation Cheezit (Deep): AAB', True):
            [
                EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB', method='proposed'),
                EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB',
                                  method='baseline_direct_edit'),
                EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB',
                                  method='baseline_rejection_sampling'),
                EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB', method='baseline_soft_rejection'),
                EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB', method='baseline_OOD_prediction'),
                EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='VAE_GAN_aab',
                                  method='VAE_GAN'),
            ],
        PlottingParams('Ablations for Simulation Cheezit (Deep): AAB', True):
            [
                EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB', method='proposed'),
                EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB',
                                  method='baseline_ignore_latent_prior'),
                EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB',
                                  method='baseline_accept_failed_projections'),
            ],

        PlottingParams('Baselines for Simulation Pitcher: YCB', True):
            [
                EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB', method='proposed'),
                EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB',
                                  method='baseline_direct_edit'),
                EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB',
                                  method='baseline_rejection_sampling'),
                EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB',
                                  method='baseline_soft_rejection'),
                EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB',
                                  method='baseline_OOD_prediction'),
                EvaluationDetails(scene_type=scenes.SimulationPitcher, network='VAE_GAN_YCB',
                                  method='VAE_GAN'),
            ],
        PlottingParams('Ablations for Simulation Pitcher: YCB', True):
            [
                EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB', method='proposed'),
                EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB',
                                  method='baseline_ignore_latent_prior'),
                EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB',
                                  method='baseline_accept_failed_projections'),
            ],

        PlottingParams('Baselines for Simulation Mug: Shapenet mugs', True):
            [
                EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs', method='proposed'),
                EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs',
                                  method='baseline_direct_edit'),
                EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs',
                                  method='baseline_rejection_sampling'),
                EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs',
                                  method='baseline_soft_rejection'),
                EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs',
                                  method='baseline_OOD_prediction'),
                EvaluationDetails(scene_type=scenes.SimulationMug, network='VAE_GAN_mugs',
                                  method='VAE_GAN'),
            ],
        PlottingParams('Ablations for Simulation Mug: Shapenet mugs', True):
            [
                EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs', method='proposed'),
                EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs',
                                  method='baseline_ignore_latent_prior'),
                EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs',
                                  method='baseline_accept_failed_projections'),
            ],

        PlottingParams('Baselines for Live Cheezit: YCB', True):
            [
                EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB', method='proposed'),
                EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB',
                                  method='baseline_direct_edit'),
                EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB',
                                  method='baseline_rejection_sampling'),
                EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB',
                                  method='baseline_soft_rejection'),
                EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB',
                                  method='baseline_OOD_prediction'),
                EvaluationDetails(scene_type=scenes.LiveScene1, network='VAE_GAN_YCB',
                                  method='VAE_GAN'),
            ],
        PlottingParams('Ablations for Live Cheezit: YCB', True):
            [
                EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB', method='proposed'),
                EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB',
                                  method='baseline_ignore_latent_prior'),
                EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB',
                                  method='baseline_accept_failed_projections'),
            ],

        PlottingParams('Baselines for Live Pitcher: YCB', True):
            [
                EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB', method='proposed'),
                EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB',
                                  method='baseline_direct_edit'),
                EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB',
                                  method='baseline_rejection_sampling'),
                EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB',
                                  method='baseline_soft_rejection'),
                EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB',
                                  method='baseline_OOD_prediction'),
                EvaluationDetails(scene_type=scenes.LivePitcher, network='VAE_GAN_YCB',
                                  method='VAE_GAN'),
            ],
        PlottingParams('Ablations for Live Pitcher: YCB', True):
            [
                EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB', method='proposed'),
                EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB',
                                  method='baseline_ignore_latent_prior'),
                EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB',
                                  method='baseline_accept_failed_projections'),
            ],


        # Skin Trials
        PlottingParams('With Skin Simulation Cheezit (Shallow): AAB', True):
            [
                EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='proposed'),
                EvaluationDetails(scene_type=scenes.SimulationCheezit, network='AAB', method='proposed_with_skin'),
            ],
        PlottingParams('With Skin Simulation Cheezit (Deep): AAB', True):
            [
                EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB', method='proposed'),
                EvaluationDetails(scene_type=scenes.SimulationDeepCheezit, network='AAB', method='proposed_with_skin'),
            ],
        PlottingParams('With Skin Simulation Pitcher: YCB', True):
            [
                EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB', method='proposed'),
                EvaluationDetails(scene_type=scenes.SimulationPitcher, network='YCB', method='proposed_with_skin'),
            ],
        PlottingParams('With Skin Simulation Mug: Shapenet mugs', True):
            [
                EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs', method='proposed'),
                EvaluationDetails(scene_type=scenes.SimulationMug, network='shapenet_mugs', method='proposed_with_skin'),
            ],
        PlottingParams('With Skin Live Cheezit: YCB', True):
            [
                EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB', method='proposed'),
                EvaluationDetails(scene_type=scenes.LiveScene1, network='YCB', method='proposed_with_skin'),
            ],
        PlottingParams('Baselines for Live Pitcher: YCB', True):
            [
                EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB', method='proposed'),
                EvaluationDetails(scene_type=scenes.LivePitcher, network='YCB', method='proposed_with_skin'),
            ],

    }
    return d


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
    parser.add_argument('--regenerate', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot_likelihood', action='store_true')
    parser.add_argument('--skip_generation', action='store_true')
    return parser.parse_args()


def generate_evaluation(details):
    tf.random.set_seed(20210108)
    scene = details.scene_type()

    contact_shape_completer = ContactShapeCompleter(scene, lookup_trial(details.network),
                                                    completion_density=1,
                                                    method=details.method)
    contact_shape_completer.get_visible_vg(load=True)

    columns = ['request number', 'chs count', 'particle num', 'scene', 'method', 'chamfer distance',
               'projection succeeded?']
    df = pd.DataFrame(columns=columns)

    gt = scene.get_gt(density_factor=1)
    point_pub = ros_helpers.get_connected_publisher('/pointcloud', PointCloud2, queue_size=1)
    point_pub.publish(gt)

    files = sorted([f for f in scene.get_save_path().glob('req_*')])
    for req_number, file in enumerate(files):
        print(f"{Fore.CYAN}Loading stored request{file}{Fore.RESET}")
        completion_req = CompleteShapeRequest()
        with file.open('rb') as f:
            completion_req.deserialize(f.read())

        completion_req.num_samples = NUM_PARTICLES_IN_TRIAL
        resp = contact_shape_completer.complete_shape_srv(completion_req)
        dists = []
        for particle_num, completion_pts_msg in enumerate(resp.sampled_completions):
            total_dist = 0
            for view in contact_shape_completer.robot_views:
                dist = vg_chamfer_distance(contact_shape_completer.transform_from_gpuvoxels(view, completion_pts_msg),
                                           contact_shape_completer.transform_from_gpuvoxels(view, gt),
                                           scale=view.scale).numpy()
                print(f"Errors w.r.t. gt: {dist}")
                total_dist += dist
            dists.append(total_dist)
            df = df.append(pd.Series([
                req_number, float(len(completion_req.chss)), particle_num, scene.name, details.method, dist, True
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


def plot(group: Tuple[PlottingParams, List[EvaluationDetails]]):
    y_label = "Chamfer Distance (cm)"
    x_label = "Num Observations {Num Total Contacts}"

    grouped_dfs = dict()
    for details in group[1]:
        scene = details.scene_type()
        df = pd.read_csv(get_evaluation_path(details))
        grouped_dfs[details.method] = df

    df = pd.concat(grouped_dfs)

    def make_x_ind(arg):
        vals = [str(int(a)) for a in arg]
        return f'{vals[0]} {{{vals[1]}}}'

    df[x_label] = df[['request number', 'chs count']].agg(make_x_ind, axis=1)
    df.rename(columns={'chamfer distance': y_label}, inplace=True)
    df['method'].replace(display_names_map, inplace=True)
    df[y_label] = 100 * df[y_label]

    display_name = display_names_map[scene.name]

    if display_name in observations_not_displayed:
        remove_observations = lambda x: x not in observations_not_displayed[display_name]
        df = df[df['request number'].map(remove_observations)]

    ax = sns.boxplot(x=x_label, y=y_label, hue='method', data=df,
                     showfliers=False)
    ax.set_title(f'{group[0].name}')

    plt.savefig(f'{root_save_path}/{group[0].name.replace(" ", "_")}')

    # if display_name not in display_legends_for:
    # if not group[0].show_legend:
    ax._remove_legend(ax.legend())

    plt.savefig(f'{root_save_path}/{group[0].name.replace(" ", "_") + "_no_legend"}')
    # plt.show()
    plt.clf()


def plot_likelihood(group: List[EvaluationDetails]):
    y_label = "Chamfer Distance (cm)"
    x_label = "Num Observations {Num Total Contacts}"

    grouped_dfs = dict()
    for details in group:
        scene = details.scene_type()
        df = pd.read_csv(get_evaluation_path(details))
        grouped_dfs[details.method] = df
    df = pd.concat(grouped_dfs)

    def make_x_ind(arg):
        vals = [str(int(a)) for a in arg]
        return f'{vals[0]} {{{vals[1]}}}'

    df[x_label] = df[['request number', 'chs count']].agg(make_x_ind, axis=1)
    df.rename(columns={'chamfer distance': y_label}, inplace=True)
    df['method'].replace(display_names_map, inplace=True)
    df[y_label] = 100 * df[y_label]

    display_name = display_names_map[scene.name]

    # Apply kernel
    def kernel(arg):
        vals = [1 / v for v in arg]
        return np.mean(vals)

    new_y_label = "Likelihood of True Scene"
    df = df[[x_label, y_label, 'method']].groupby(['method', x_label]).agg({y_label: kernel,
                                                                            'method': 'first',
                                                                            x_label: 'first'})

    df.rename(columns={y_label: new_y_label}, inplace=True)
    df['linesize'] = df['method'].agg(lambda x: linesize[x])

    ax = sns.lineplot(x=x_label, y=new_y_label, hue='method', data=df, size='linesize')
    ax.set_title(f'{display_name}: {group[0].network}')


    if display_name not in display_legends_for:
        ax._remove_legend(ax.legend())
    plt.savefig(f'{root_save_path}/{scene.name}_prob_score')
    plt.show()


def main():
    ARGS = parse_command_line_args()

    rospy.init_node('contact_completion_evaluation')
    rospy.loginfo("Contact Completion Evaluation")

    if not ARGS.skip_generation:
        for groups in get_evaluation_trial_groups().values():
            for details in groups:
                if ARGS.regenerate:
                    print(f'{Fore.CYAN}Regenerating {details}{Fore.RESET}')
                    generate_evaluation(details)
                elif not get_evaluation_path(details).exists():
                    print(f'{Fore.CYAN}Generating {details}{Fore.RESET}')
                    generate_evaluation(details)
                else:
                    print(f"{details} exists. Not generating")

    if ARGS.plot:
        for group in get_evaluation_trial_groups().items():
            plot(group)
    if ARGS.plot_likelihood:
        for group in get_evaluation_trial_groups().items():
            plot_likelihood(group)


if __name__ == "__main__":
    main()
