#! /usr/bin/env python
import shape_completion_training.utils.old_dataset_tools
from shape_completion_training.utils import dataset_supervisor

if __name__ == "__main__":
    shape_completion_training.utils.old_dataset_tools.write_shapenet_to_filelist(test_ratio=0.15,
                                                                                 shape_ids=dataset_supervisor.shapenet_labels(["table"]))
