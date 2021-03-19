#! /usr/bin/env python
import argparse

import shape_completion_training.utils.old_dataset_tools
from shape_completion_training.utils import shapenet_storage
from shape_completion_training.utils import data_tools
from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.model import default_params

override_params = {
    "use_flow_during_inference": True
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args for training")
    parser.add_argument('--tmp', action='store_true')
    parser.add_argument('--group', default=None)
    args = parser.parse_args()
    params = default_params.get_default_params(group_name=args.group)
    params.update(override_params)

    # data, _ = shape_completion_training.utils.old_dataset_tools.load_dataset(params['dataset'], metadata_only=False)
    # data = shape_completion_training.utils.old_dataset_tools.preprocess_dataset(data, params)
    data_supervisor = shapenet_storage.ShapenetDatasetSupervisor(params['dataset'])

    if args.tmp:
        mr = ModelRunner(training=True, params=params, group_name=None)
    else:
        mr = ModelRunner(training=True, params=params, group_name=args.group)

    mr.train_and_test(data_supervisor)
