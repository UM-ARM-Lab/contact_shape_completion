#! /usr/bin/env python
from __future__ import print_function

import shape_completion_training.utils.shapenet_storage
from shape_completion_training.utils import shapenet_storage
import datetime
import rospy
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

# def augment_single(basepath):
#     """
#     Augment a hardcoded single shape. Useful for debugging
#     """
#     shape_id = 'a1d293f5cc20d01ad7f470ee20dce9e0'
#     fp = basepath / shape_id / 'models'
#     print("Augmenting single models at {}".format(fp))
#
#     old_files = [f for f in fp.iterdir() if f.name.startswith("model_augmented")]
#     for f in old_files:
#         f.unlink()
#
#     obj_path = fp / "model_normalized.obj"
#     obj_tools.augment(obj_path.as_posix())
#
#     augmented_obj_files = [f for f in fp.iterdir()
#                            if f.name.startswith('model_augmented')
#                            if f.name.endswith('.obj')]
#     augmented_obj_files.sort()
#     for f in augmented_obj_files:
#         binvox_object_file(f)


if __name__ == "__main__":
    rospy.init_node("augment_shapenet_node")
    sn_path = shapenet_storage.shapenet_load_path
    # sn_path = sn_path / shape_completion_training.utils.shapenet_storage.shape_map['mug']
    sn_path = sn_path / shape_completion_training.utils.shapenet_storage.shape_map['table']

    start_time = datetime.datetime.now()

    # augment_category(sn_path)
    augment_category(sn_path, shape_ids=["ff5a2e340869e9c45981503fc6dfccb2",
                                         "ff60e4b29c5cc38fceda3ac62a593e9c",
                                         "ff9bb9597cac5ef17a50afc9c93f8a50",
                                         # "ffa71bb0a75ebd7f93ad7cfe5cf8e21f",
                                         # "ffa875f5e2242d62d13de1e342854aa",
                                         # "ffb5e48fde2cca54518bdb78540c51ed",
                                         # "ffb7b155cea1159a3a8e4d3441f2dd18",
                                         # "ffc2c7813c80d8fd323d6bd8db8de5b",
                                         # "ffc75a8bdb88c751b7fbcb21c074906c",
                                         # "ffd45a17e325bfa91933ffef19678834",
                                         "ffe02f7b3b421ee96cff9b44fdf0517e",
                                         "ffe1c487f7b9909bfebad4f49b26ec52",
                                         "ffe2bf44f5d9760b9a8ef44e1d2c5b75",
                                         "ffe4383cff6d000a3628187d1bb97b92",
                                         "fff492e352c8cb336240c88cd4684446",
                                         "fff7f07d1c4042f8a946c24c4f9fb58e"
                                         ])
    print("")
    print("Augmenting with {} threads took {} seconds".format(NUM_THREADS, datetime.datetime.now() - start_time))
