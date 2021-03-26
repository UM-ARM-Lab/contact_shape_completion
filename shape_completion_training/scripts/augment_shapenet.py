#! /usr/bin/env python
from __future__ import print_function

import argparse

import shape_completion_training.utils.shapenet_storage
from shape_completion_training.utils import shapenet_storage
import datetime
import rospy
from shape_completion_training.utils.config import get_config
from shape_completion_training.utils.data_augmentation import NUM_THREADS, augment_category

# HARDCODED_BOUNDARY = '-bb -0.6 -0.6 -0.6 0.6 0.6 0.6'

"""
NOTE:
If running over ssh, need to start a virtual screen
https://www.patrickmin.com/binvox/

Xvfb :99 -screen 0 640x480x24 &
export DISPLAY=:99


Then run binvox with the -pb option 

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reverse',
                        help="reverses the order of categories when augmenting. "
                             "Useful when running on two computers",
                        action='store_true')
    parser.add_argument('--fast', action="store_true", help="Skip files that have already been processed")
    args = parser.parse_args()

    ds_path = shapenet_storage.get_shapenet_path()
    shape_map = shapenet_storage.get_shape_map()

    categories = list(shape_map.keys())
    categories.remove('car')

    category = ['table']
    if args.reverse:
        categories.reverse()

    for i, category in enumerate(categories):
        cat_id = shape_map[category]
        print(f"\n\nAugmenting {category} ({cat_id}). {i + 1}/{len(categories)}")
        start_time = datetime.datetime.now()

        shape_ids = None

        augment_category(ds_path, cat_id, shape_ids=shape_ids, only_new_files=args.fast)

        print("")
        print("Augmenting with {} threads took {} seconds".format(NUM_THREADS, datetime.datetime.now() - start_time))
