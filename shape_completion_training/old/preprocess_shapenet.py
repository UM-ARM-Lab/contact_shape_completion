#! /usr/bin/env python
import shape_completion_training.utils.old_dataset_tools
from shape_completion_training.utils import shapenet_storage

if __name__ == "__main__":
    shape_completion_training.utils.old_dataset_tools.write_shapenet_to_filelist(test_ratio=0.15,
                                                                                 shape_ids=shapenet_storage.shapenet_labels(["table"]))
