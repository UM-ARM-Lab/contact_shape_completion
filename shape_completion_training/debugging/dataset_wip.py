#! /usr/bin/env python
"""
Debugging the custom dataset supervisor class

"""
import datetime
import math
import time
import tensorflow as tf

import progressbar

from shape_completion_training.model import default_params
from shape_completion_training.utils import dataset_supervisor
from shape_completion_training.utils.dataset_supervisor import ShapenetDatasetSupervisor

shapenet_categories_for = {
    "shapenet_mugs": ['mug'],
    "shapenet_airplanes": ['airplane'],
    "shapenet_tables": ['table'],
    "shapenet_bag": ['bag']}


def create_shapenet_only_datasets():
    for name, categories in shapenet_categories_for.items():
        ds = ShapenetDatasetSupervisor(name, require_exists=False)
        # if ds.get_save_path().exists():
        #     print(f"Dataset {name} already exists")
        #     continue
        print(f"Creating dataset {name}...")
        fps = [dataset_supervisor.get_shapenet_map()[c] for c in categories]
        ds.create_new_dataset(fps)
        # ds.save()
        print(f"Saved Dataset {name}")


def load_ds():
    data_supervisor = dataset_supervisor.ShapenetDatasetSupervisor('shapenet_mugs')
    training = data_supervisor.get_training(default_params.get_visualization_params())
    elem = next(training.batch(1))
    print(f"The filepath stored in the first element is: {elem.md[0]['filepath']}")
    elem.load()


def load_full_ds_in_batches():
    params = default_params.get_default_params(group_name='PSSNet')
    data_supervisor = dataset_supervisor.ShapenetDatasetSupervisor('shapenet_mugs')
    training = data_supervisor.get_training(params)
    batch_size = params['batch_size']
    num_batches = math.ceil(training.size() / batch_size)

    widgets = [
        '  ', progressbar.Counter(), '/', str(num_batches),
        ' ', progressbar.Variable("Loss"), ' ',
        progressbar.Bar(),
        ' [', progressbar.Variable("TrainTime"), '] ',
        ' (', progressbar.ETA(), ') ',
    ]

    training_batches = training.batch(params['batch_size'])

    with progressbar.ProgressBar(widgets=widgets, max_value=num_batches) as bar:
        t0 = time.time()
        batch_num = 0
        for batch in training_batches:
            batch_num += 1
            # self.ckpt.step.assign_add(1)
            data = batch.load()

            # _, ret = self.model.train_step(data)
            time_str = str(datetime.timedelta(seconds=int(time.time() - t0)))
            bar.update(batch_num, Loss=0,
                       TrainTime=time_str)
            # if self.num_batches % summary_period == 0:
            #     self.write_summary(ret)
            # self.ckpt.train_time.assign_add(time.time() - t0)
            # t0 = time.time()


def multiple_loading_of_batches(num_loadings=100):
    for i in range(num_loadings):
        load_full_ds_in_batches()
        print(f"Loaded dataset {i + 1} times")


def check_load_bounding_box_only():
    data_supervisor = dataset_supervisor.get_dataset_supervisor('ycb_all')
    dataset = data_supervisor.get_training(params=default_params.get_visualization_params())
    for batch in progressbar.progressbar(dataset.batch(10)):
        elem = batch.load_bounding_box_only()


def main():
    # create_shapenet_only_datasets()
    # load_ds()
    # multiple_loading_of_batches()
    check_load_bounding_box_only()


if __name__ == "__main__":
    main()
