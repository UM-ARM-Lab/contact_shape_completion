#! /usr/bin/env python
import shape_completion_training.utils.old_dataset_tools
import shape_completion_training.utils.dataset_supervisor
from shape_completion_training.utils import data_tools


def process(ds):
    names = set()
    num = 0
    for elem in ds:
        names.add(elem['id'].numpy())
        num += 1
    return {"names": names, "num_elements": num}


def report(ds_name, info):
    print("{} has:".format(ds_name))
    for k, v in info.items():
        print("{}: {}".format(k, v))


def write_all_files():
    with open("./file_names.txt", "w") as f:
        for record in shape_completion_training.utils.dataset_supervisor.get_all_shapenet_files(shape_ids=shape_completion_training.utils.dataset_supervisor.shapenet_labels(["mug"])):
            f.write("{}\n".format(record.id))



if __name__ == "__main__":
    train_ds, test_ds = shape_completion_training.utils.old_dataset_tools.load_shapenet_metadata()
    print(report("test dataset", process(test_ds)))
    # write_all_files()

