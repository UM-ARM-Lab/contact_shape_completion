#!/usr/bin/env python
from __future__ import print_function

import argparse

import rospy

from contact_shape_completion.contact_shape_completer import ContactShapeCompleter
from contact_shape_completion.goal_generator import CheezeitGoalGenerator
from contact_shape_completion.scenes import SimulationCheezit, get_scene
from shape_completion_training.model import default_params
from shape_completion_training.utils.config import lookup_trial

"""
Publish object pointclouds for use in gpu_voxels planning
"""

default_dataset_params = default_params.get_default_params()


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
    parser.add_argument('--trial', required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--store', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_command_line_args()

    rospy.init_node('contact_shape_completer_service')
    rospy.loginfo("Data Publisher")

    # scene = SimulationCheezit()
    scene = get_scene(ARGS.scene)

    contact_shape_completer = ContactShapeCompleter(scene, lookup_trial(ARGS.trial),
                                                    store_request=ARGS.store)
    # contact_shape_completer.load_network(ARGS.trial)

    contact_shape_completer.get_visible_vg()
    contact_shape_completer.compute_known_occ()

    print("Up and running")
    rospy.spin()
