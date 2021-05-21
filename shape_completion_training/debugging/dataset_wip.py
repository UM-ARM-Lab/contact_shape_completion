#! /usr/bin/env python
"""
Debugging the custom dataset supervisor class

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
        # if ds.get_save_path().exists():
        #     print(f"Dataset {name} already exists")
        #     continue
        print(f"Creating dataset {name}...")
        fps = [shapenet_storage.get_shapenet_map()[c] for c in categories]
        ds.create_new_dataset(fps)
        # ds.save()
        print(f"Saved Dataset {name}")

def load_ds():
    data_supervisor = shapenet_storage.ShapenetDatasetSupervisor('shapenet_mugs')
    training = data_supervisor.get_training()
    elem = next(training.batch(1))
    print(f"The filepath stored in the first element is: {elem.md[0]['filepath']}")
    elem.load()

def main():
    # create_shapenet_only_datasets()
    load_ds()


if __name__ == "__main__":
    main()
