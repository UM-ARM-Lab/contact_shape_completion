#! /usr/bin/env python
import shape_completion_training.utils.dataset_loader
import shape_completion_training.utils.old_dataset_tools
import shape_completion_training.utils.dataset_supervisor
from shape_completion_training.utils import data_tools, dataset_supervisor
from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.model import default_params
import argparse

# params = {
#     'batch_size': 1500,
#     'network': 'RealNVP',
#     'dim': 24,
#     'num_masked': 12,
#     'learning_rate': 1e-5,
#     'translation_pixel_range_x': 10,
#     'translation_pixel_range_y': 10,
#     'translation_pixel_range_z': 10,
# }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args for training")
    parser.add_argument('--tmp', action='store_true')
    parser.add_argument('--group', default=None)
    args = parser.parse_args()
    params = default_params.get_default_params(group_name=args.group)
    params['load_bb_only'] = True

    data_supervisor = shape_completion_training.utils.dataset_loader.get_dataset_supervisor(params['dataset'])

    if args.tmp:
        mr = ModelRunner(training=True, params=params, group_name=None)
    else:
        mr = ModelRunner(training=True, params=params, group_name=args.group)

    mr.train_and_test(data_supervisor)
