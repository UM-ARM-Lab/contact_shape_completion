#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import pwd

import rospy
from window_recorder.recorder import WindowRecorder

from contact_shape_completion.contact_shape_completer import ContactShapeCompleter
from contact_shape_completion.scenes import get_scene
from shape_completion_training.model import default_params
from shape_completion_training.utils.config import lookup_trial

"""
Publish object pointclouds for use in gpu_voxels planning
"""

YOUR_USERNAME = pwd.getpwuid(os.getuid())[0]
rviz_capture_path = f'/home/{YOUR_USERNAME}/Videos/shape_contact/captures'

default_dataset_params = default_params.get_default_params()


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
    parser.add_argument('--segment')
    return parser.parse_args()


def make_aab_video_1():
    scene = get_scene('cheezit_01')
    trial = lookup_trial('AAB')

    with WindowRecorder(["live_shape_completion.rviz* - RViz", "live_shape_completion.rviz - RViz"], frame_rate=30.0,
                        name_suffix="rviz",
                        save_dir=rviz_capture_path):
        contact_shape_completer = ContactShapeCompleter(scene, trial,
                                                        store_request=False)
        contact_shape_completer.get_visible_vg()
        contact_shape_completer.compute_known_occ()

        print("Up and running")
        rospy.spin()


def make_mug_video_1():
    scene = get_scene('mug')
    trial = lookup_trial('shapenet_mugs')

    with WindowRecorder(["live_shape_completion.rviz* - RViz", "live_shape_completion.rviz - RViz"], frame_rate=30.0,
                        name_suffix="rviz",
                        save_dir=rviz_capture_path):
        contact_shape_completer = ContactShapeCompleter(scene, trial,
                                                        store_request=False)
        contact_shape_completer.get_visible_vg()
        contact_shape_completer.compute_known_occ()

        print("Up and running")
        rospy.spin()

def make_pitcher_video_1():
    scene = get_scene('pitcher')
    trial = lookup_trial('YCB')

    with WindowRecorder(["live_shape_completion.rviz* - RViz", "live_shape_completion.rviz - RViz"], frame_rate=30.0,
                        name_suffix="rviz",
                        save_dir=rviz_capture_path):
        contact_shape_completer = ContactShapeCompleter(scene, trial,
                                                        store_request=False)
        contact_shape_completer.get_visible_vg()
        contact_shape_completer.compute_known_occ()

        print("Up and running")
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node('video_maker')
    ARGS = parse_command_line_args()

    fun_map = {'aab': make_aab_video_1,
               'mug': make_mug_video_1,
               'pitcher': make_pitcher_video_1()}
    fun_map[ARGS.segment]()
