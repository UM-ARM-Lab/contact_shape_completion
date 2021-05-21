import shape_completion_training.utils.dataset_storage
import shape_completion_training.utils.old_dataset_tools
from shape_completion_training.model import filepath_tools
from shape_completion_training.utils import data_tools, dataset_supervisor
import datetime

shapenet_load_path = dataset_supervisor.get_shapenet_path()


def store_shapenet_augmentations_in_multiple_formats(shape_ids):
    i = 0
    shapenet_records = []
    if shape_ids == "all":
        shape_ids = [f.name for f in shapenet_load_path.iterdir() if f.is_dir()]
        # shape_ids = [f for f in os.listdir(shapenet_load_path)
        #              if os.path.isdir(join(shapenet_load_path, f))]
        shape_ids.sort()

    for i in range(0, len(shape_ids)):
        category = shape_ids[i]
        shape_path = shapenet_load_path / category
        for obj_fp in sorted(p for p in shape_path.iterdir()):
            all_augmentations = [f for f in (obj_fp / "models").iterdir()
                                 if f.name.startswith("model_augmented")
                                 if f.name.endswith(".pkl")]
            for f in sorted(all_augmentations):
                i += 1
                print(i)
                # shapenet_records.append(load_gt_voxels(f))
                gt = shape_completion_training.utils.dataset_storage.load_data_with_gt(f, "gzip")
                shape_completion_training.utils.dataset_storage.save_gt_voxels(f, gt['gt_occ'], "bz2")
                shape_completion_training.utils.dataset_storage.save_gt_voxels(f, gt['gt_occ'], "gzip")
                shape_completion_training.utils.dataset_storage.save_gt_voxels(f, gt['gt_occ'], "None")

    return shapenet_records


def load_all_files_and_record_time(shape_ids, compression=None):
    start = datetime.datetime.now()
    for i in range(0, len(shape_ids)):
        category = shape_ids[i]
        shape_path = shapenet_load_path / category
        for obj_fp in sorted(p for p in shape_path.iterdir()):
            all_augmentations = [f for f in (obj_fp / "models").iterdir()
                                 if f.name.startswith("model_augmented")
                                 if f.name.endswith(".pkl")]
            for f in sorted(all_augmentations):
                i += 1
                print(i)
                # shapenet_records.append(load_gt_voxels(f))
                gt = shape_completion_training.utils.dataset_storage.load_data_with_gt(f, compression)
                # shapenet_storage.save_gt_voxels(f, gt['gt_occ'], "bz2")
                # shapenet_storage.save_gt_voxels(f, gt['gt_occ'], "gzip")
                # shapenet_storage.save_gt_voxels(f, gt['gt_occ'], None)
    print("Loading using compression {} took {}".format(compression, datetime.datetime.now() - start))


def load_shapenet():
    start = datetime.datetime.now()
    sn = shape_completion_training.utils.old_dataset_tools.get_addressible_dataset(use_train=False)
    a = sn.test_map.keys()[0]
    md = sn.get_metadata(a)
    d = sn.get(a)
    print("loading took {}".format(datetime.datetime.now() - start))


if __name__ == "__main__":
    # store_shapenet_augmentations_in_multiple_formats(["03797390"])
    # load_all_files_and_record_time(["03797390"], compression="bz2")
    # shapenet = data_tools.load_shapenet([data_tools.shape_map["mug"]])
    load_shapenet()
    # elem = next(shapenet.__iter__())
    # print(elem)
