"""
Prepares some local files of shapes to be used in unit tests
"""

import shape_completion_training.utils.data_tools as data_tools
import numpy as np

import shape_completion_training.utils.dataset_supervisor

test_dataset_filepath = "voxel_grid_test_data.npy"


def create_tests_dataset():
    train_data_shapenet, test_data_shapenet = data_tools.load_shapenet([
        shape_completion_training.utils.dataset_supervisor.get_shapenet_map()["mug"]])
    data_shapenet = train_data_shapenet.shuffle(1000)

    test_shapes = []

    it = data_shapenet.__iter__()
    for _ in range(10):
        elem = next(it)
        test_shapes.append(elem['gt_occ'].numpy())
    shapes = np.array(test_shapes)
    with open(test_dataset_filepath, "w") as file:
        np.save(file, test_shapes)
    print("done")


def load_test_files():
    # with open(test_dataset_filepath) as file:
    #     single_mat = np.load(file, encoding='latin1')
    single_mat = np.load(test_dataset_filepath, encoding='ASCII')
    return np.split(single_mat, single_mat.shape[0], axis=0)


if __name__ == "__main__":
    create_tests_dataset()
    mat = load_test_files()
