#! /usr/bin/env python
"""
This script takes shapenet (and other?) voxel datasets and compiles ShapenetDatasetSupervisors, which are
metadatasets.

"""
from shape_completion_training.utils import shapenet_storage
from shape_completion_training.utils.shapenet_storage import ShapenetDatasetSupervisor

shapenet_categories_for = {
    "shapenet_mugs": ['mug'],
    "shapenet_airplanes": ['airplane'],
    "shapenet_tables": ['table'],
    "shapenet_bag": ['bag']}


def create_shapenet_only_datasets():
    for name, categories in shapenet_categories_for.items():
        ds = ShapenetDatasetSupervisor(name, require_exists=False)
        if ds.get_save_path().exists():
            print(f"Dataset {name} already exists")
            continue
        print(f"Creating dataset {name}...")
        fps = [shapenet_storage.get_shape_map()[c] for c in categories]
        ds.create_new_dataset(fps)
        ds.save()
        print(f"Saved Dataset {name}")


def main():
    create_shapenet_only_datasets()


if __name__ == "__main__":
    main()
