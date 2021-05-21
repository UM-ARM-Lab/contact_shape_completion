#! /usr/bin/env python
from __future__ import print_function

import shape_completion_training.utils.dataset_supervisor
from shape_completion_training.utils import dataset_supervisor
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
#                            if f.name.startswith('binmodel_augmented')
#                            if f.name.endswith('.obj')]
#     augmented_obj_files.sort()
#     for f in augmented_obj_files:
#         binvox_object_file(f)


if __name__ == "__main__":
    # rospy.init_node("augment_shapenet_node")
    ds_path = dataset_supervisor.get_shapenet_path()
    category = dataset_supervisor.get_shapenet_map()['table']
    # category = shapenet_storage.get_shape_map()['mug']

    start_time = datetime.datetime.now()

    # shape_ids = ["10155655850468db78d106ce0a280f87",
    #              "1021a0914a7207aff927ed529ad90a11",
    #              "1026dd1b26120799107f68a9cb8e3c",
    #              "103c9e43cdf6501c62b600da24e0965",
    #              "105f7f51e4140ee4b6b87e72ead132ed",
    #              ]
    shape_ids = None

    # augment_category(sn_path)
    augment_category(ds_path, category, shape_ids=shape_ids)

    # sup = shapenet_storage.ShapenetDatasetSupervisor("shapenet_wip_mugs")
    # ds.create_new_dataset([category])
    # ds = sup.get_training()
    # a = next(ds.batch(10))
    # data = a.load()
    print("")
    print("Augmenting with {} threads took {} seconds".format(NUM_THREADS, datetime.datetime.now() - start_time))
