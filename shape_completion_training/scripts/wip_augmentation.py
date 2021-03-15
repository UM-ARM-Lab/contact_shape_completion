# import pymesh
from shape_completion_training.utils import dataset_storage

def main():
    path = dataset_storage.package_path / "data/ShapeNetCore.v2_augmented/03797390/1a1c0a8d4bad82169f0594e65f756cf5/models"
    mesh = pymesh.load_mesh(path / "model_normalized.obj")


if __name__ == "__main__":
    main()