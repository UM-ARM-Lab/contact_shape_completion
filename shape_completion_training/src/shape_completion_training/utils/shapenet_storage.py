from shape_completion_training.model import filepath_tools
from shape_completion_training.utils import tf_utils
from shape_completion_training.utils.dataset_storage import load_metadata, _split_train_and_test, write_to_filelist
import hjson
from pathlib import Path

"""
Tools for storing and preprocessing augmented shapenet
"""


def write_shapenet_to_filelist(test_ratio, shape_ids="all"):
    all_files = get_all_shapenet_files(shape_ids)
    train_files, test_files = _split_train_and_test(all_files, test_ratio)
    # train_data = _list_of_shapenet_records_to_dict(train_files)
    # test_data = _list_of_shapenet_records_to_dict(test_files)

    print("Split shapenet into {} training and {} test shapes".format(len(train_files), len(test_files)))

    # d = tf.data.Dataset.from_tensor_slices(utils.sequence_of_dicts_to_dict_of_sequences(test_files))
    write_to_filelist(tf_utils.sequence_of_dicts_to_dict_of_sequences(train_files),
                      shapenet_record_path / "train_filepaths.pkl")
    write_to_filelist(tf_utils.sequence_of_dicts_to_dict_of_sequences(test_files),
                      shapenet_record_path / "test_filepaths.pkl")
    # write_to_tfrecord(tf.data.Dataset.from_tensor_slices(
    #     utils.sequence_of_dicts_to_dict_of_sequences(test_files)),
    #     shapenet_record_path / "test_filepaths.pkl")


def get_all_shapenet_files(shape_ids):
    shapenet_records = []
    if shape_ids == "all":
        shape_ids = [f.name for f in shapenet_load_path.iterdir() if f.is_dir()]
        # shape_ids = [f for f in os.listdir(shapenet_load_path)
        #              if os.path.isdir(join(shapenet_load_path, f))]
        shape_ids.sort()

    for category in shape_ids:
        shape_path = shapenet_load_path / category
        for obj_fp in sorted(p for p in shape_path.iterdir()):
            print("{}".format(obj_fp.name))
            all_augmentations = [f for f in (obj_fp / "models").iterdir()
                                 if f.name.startswith("model_augmented")
                                 if f.name.endswith(".pkl.gzip")]
            for f in sorted(all_augmentations):
                # shapenet_records.append(load_gt_voxels(f))
                base = f.parent / f.stem
                shapenet_records.append(load_metadata(base, compression="gzip"))

    return shapenet_records


def get_shapenet_path():
    fp = filepath_tools.get_shape_completion_package_path() / "config.hjson"
    with fp.open() as f:
        config = hjson.load(f)

    if 'shapenet_path' not in config:
        raise KeyError("shapenet_path must be defined in config.hjson file")

    p = Path(config['shapenet_path'])
    if p.is_absolute():
        return p

    return filepath_tools.get_shape_completion_package_path() / p


def get_shape_map():
    sn_path = get_shapenet_path()
    with (sn_path / "taxonomy.json").open() as f:
        taxonomy = hjson.load(f)

    sm = dict()
    categories = [d.parts[-1] for d in sn_path.iterdir()]
    for t in taxonomy:
        if t['synsetId'] in categories:
            name = t['name'].split(',')[0]
            sm[name] = t['synsetId']

    return sm


# shapenet_load_path = filepath_tools.get_shape_completion_package_path() / "data" / "ShapeNetCore.v2_multirotation"
# shapenet_load_path = filepath_tools.get_shape_completion_package_path() / "data" / "ShapeNetCore.v2_augmented"
shapenet_load_path = get_shapenet_path()
shapenet_record_path = shapenet_load_path / "tfrecords" / "filepath"
# shape_map = {"airplane": "02691156",
#              "mug": "03797390",
#              "table": "04379243"}
shape_map = get_shape_map()


def shapenet_labels(human_names):
    return [shape_map[hn] for hn in human_names]