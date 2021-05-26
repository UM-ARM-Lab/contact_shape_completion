#!/usr/bin/env python
from __future__ import print_function

import argparse

import rospy
from gpu_voxel_planning_msgs.srv import CompleteShapeRequest

from contact_shape_completion.contact_shape_completer import ContactShapeCompleter
from contact_shape_completion.goal_generator import CheezeitGoalGenerator
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
    contact_shape_completer = ContactShapeCompleter(ARGS.trial, goal_generator=goal_generator)
    # contact_shape_completer.load_network(ARGS.trial)

    contact_shape_completer.load_visible_vg(filename='wip_segmented_pts.msg')
    completion_req = CompleteShapeRequest()
    with (contact_shape_completer.get_wip_save_path() / 'wip_req.msg').open('rb') as f:
        completion_req.deserialize(f.read())

    contact_shape_completer.complete_shape_srv(completion_req)

    # contact_shape_completer.compute_known_occ()
    # contact_shape_completer.save_last_visible_vg()
    # contact_shape_completer.load_last_visible_vg()

    # contact_shape_completer.infer_completion()
    # contact_shape_completer.do_some_completions_debug()

    # contact_shape_completer.infer_completion()

    # print("Up and running")
    # rospy.spin()
