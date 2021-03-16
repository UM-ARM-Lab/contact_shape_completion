from shape_completion_training.model import filepath_tools
from shape_completion_training.utils import tf_utils
from shape_completion_training.utils.dataset_storage import load_metadata, _split_train_and_test, write_to_filelist
import hjson
from pathlib import Path
from functools import lru_cache
from itertools import chain
import pickle

"""
Tools for storing and preprocessing augmented shapenet
"""


class ShapenetMetaDataset:
    def __init__(self, name):
        self.name = name
        self.train_md = None
        self.test_md = None

        self.ind_for_train_id = dict()
        self.ind_for_test_id = dict()

    def create_new_dataset(self, shape_ids, test_ratio=0.1):
        files = get_all_shapenet_files(shape_ids)
        train_files, test_files = _split_train_and_test(files, test_ratio)
        self.train_md = train_files
        self.test_md = test_files

        for i, elem in enumerate(self.train_md):
            self.ind_for_train_id[get_unique_name(elem)] = i
        for i, elem in enumerate(self.test_md):
            self.ind_for_test_id[get_unique_name(elem)] = i


    def save(self, overwrite=False):
        fp = get_shapenet_path() / "MetaDataSets"
        fp.mkdir(exist_ok=True)
        fp = fp / f'{self.name}.metadataset.pkl'
        if fp.exists() and not overwrite:
            raise FileExistsError(f"Metadataset already exists {fp.as_posix()}")

        with fp.open('wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self):
        fp = get_shapenet_path() / "MetaDataSets" / f'{self.name}.metadataset.pkl'
        if not fp.exists():
            raise FileNotFoundError(f"metadataset file not found: {fp.as_posix()}")
        with fp.open('rb') as f:
            self.__dict__ = pickle.load(f)


def get_unique_name(datum, has_batch_dim=False):
    """
    Returns a unique name for the datum
    @param datum:
    @return:
    """
    # if has_batch_dim:
    #     return (datum['id']+ datum['augmentation'])
    return datum['id'] + datum['augmentation']


def write_shapenet_to_filelist(test_ratio, shape_ids="all"):
    all_files = get_all_shapenet_files(shape_ids)
    train_files, test_files = _split_train_and_test(all_files, test_ratio)
    # train_data = _list_of_shapenet_records_to_dict(train_files)
    # test_data = _list_of_shapenet_records_to_dict(test_files)

    print("Split shapenet into {} training and {} test shapes".format(len(train_files), len(test_files)))

    # d = tf.data.Dataset.from_tensor_slices(utils.sequence_of_dicts_to_dict_of_sequences(test_files))
    write_to_filelist(tf_utils.sequence_of_dicts_to_dict_of_sequences(train_files),
                      get_shapenet_record_path / "train_filepaths.pkl")
    write_to_filelist(tf_utils.sequence_of_dicts_to_dict_of_sequences(test_files),
                      get_shapenet_record_path / "test_filepaths.pkl")
    # write_to_tfrecord(tf.data.Dataset.from_tensor_slices(
    #     utils.sequence_of_dicts_to_dict_of_sequences(test_files)),
    #     shapenet_record_path / "test_filepaths.pkl")


def get_all_shapenet_files(shape_ids):
    shapenet_records = []
    if shape_ids == "all":
        shape_ids = [f.name for f in get_shapenet_path().iterdir() if f.is_dir()]
        # shape_ids = [f for f in os.listdir(shapenet_load_path)
        #              if os.path.isdir(join(shapenet_load_path, f))]
        shape_ids.sort()

    for category in shape_ids:
        shape_path = get_shapenet_path() / category
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


@lru_cache()
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


@lru_cache()
def get_shapenet_record_path():
    return get_shapenet_path() / "tfrecords" / "filepath"


@lru_cache()
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


def shapenet_labels(human_names):
    return [get_shape_map()[hn] for hn in human_names]
