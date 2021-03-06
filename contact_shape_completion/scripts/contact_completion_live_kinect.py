#!/usr/bin/env python
from __future__ import print_function

import argparse

import rospy

from contact_shape_completion.contact_shape_completer import ContactShapeCompleter
from contact_shape_completion.goal_generator import CheezeitGoalGenerator
from contact_shape_completion.scenes import LiveScene1
from shape_completion_training.model import default_params

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
    scene = LiveScene1()
    contact_shape_completer = ContactShapeCompleter(scene, ARGS.trial, goal_generator=goal_generator)
    # contact_shape_completer.load_network(ARGS.trial)

    contact_shape_completer.get_visible_vg()
    contact_shape_completer.compute_known_occ()
    # contact_shape_completer.save_last_visible_vg()
    # contact_shape_completer.load_last_visible_vg()

    # contact_shape_completer.infer_completion()
    # contact_shape_completer.do_some_completions_debug()

    # contact_shape_completer.infer_completion()

    print("Up and running")
    rospy.spin()
