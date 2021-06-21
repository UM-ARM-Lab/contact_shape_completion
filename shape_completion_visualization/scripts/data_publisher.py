#!/usr/bin/env python

import rospy

from shape_completion_training.model import default_params
from shape_completion_visualization.visualizer import Visualizer, parse_visualizer_command_line_args


default_dataset_params = default_params.get_default_params()

overwrite_params = {
    'translation_pixel_range_x': 10,
    'translation_pixel_range_y': 10,
    'translation_pixel_range_z': 10,
    'apply_depth_sensor_noise': False,
    'apply_slit_occlusion': False
}

if __name__ == "__main__":
    rospy.init_node('shape_publisher')
    rospy.loginfo("Data Publisher")
    args = vars(parse_visualizer_command_line_args())
    params = default_params.get_visualization_params()
    params.update(overwrite_params)
    print(f"Using params {params}")
    visualizer = Visualizer(**args, params=params)

    rospy.spin()
