import abc
import pickle
from functools import lru_cache
from pathlib import Path

import hjson

from shape_completion_training.model import filepath_tools
from shape_completion_training.utils.config import get_config
from shape_completion_training.utils.data_tools import simulate_2_5D_input
from shape_completion_training.utils.dataset_storage import load_metadata, _split_train_and_test, load_gt_only
from shape_completion_training.utils.tf_utils import sequence_of_dicts_to_dict_of_sequences, stack_dict

"""
Tools for storing and preprocessing augmented shapenet
"""


class MetaDataset(abc.ABC):
    load_limit = 20

    def __init__(self, metadata):
        self.md = metadata

    @abc.abstractmethod
    def absolute_fp(self, rel_fp):
        pass

    def batch(self, batch_size):
        for i in range(0, len(self.md), batch_size):
            yield self.__class__(self.md[i:i + batch_size])

    def load(self):
        if len(self.md) > self.load_limit:
            raise OverflowError(f"Asked to load {len(self.md)} shapes. Too large. You probably didnt mean that")
        for elem in self.md:
            rel_fp = elem['filepath']
            # TODO: This fixes a temporary bug with the way the filepath is saved if the shapenet path is relative,
            #  (not absolute)
            fp = self.absolute_fp(rel_fp)
            vg = load_gt_only(fp)
            elem['gt_occ'] = vg
            elem['gt_free'] = 1 - vg
            ko, kf = simulate_2_5D_input(vg)
            elem['known_occ'] = ko
            elem['known_free'] = kf
            # print(f"Loading {elem['filepath']}")
        return stack_dict(sequence_of_dicts_to_dict_of_sequences(self.md))


class ShapenetMetaDataset(MetaDataset):
    def __init__(self, metadata):
        super().__init__(metadata)

    def absolute_fp(self, rel_fp):
        prefix = "data/ShapeNetCore.v2_augmented/"
        if rel_fp.startswith(prefix):
            rel_fp = rel_fp[len(prefix):]
        fp = get_shapenet_path() / rel_fp
        return fp


class YcbMetaDataset(MetaDataset):
    def __init__(self, metadata):
        super().__init__(metadata)

    def absolute_fp(self, rel_fp):
        return get_ycb_path() / rel_fp


class DatasetSupervisor(abc.ABC):
    def __init__(self, name, meta_dataset_type, require_exists=True, load=True):
        self.meta_dataset_type = meta_dataset_type
        self.name = name
        self.train_md = None
        self.test_md = None

        self.ind_for_train_id = dict()
        self.ind_for_test_id = dict()

        if load:
            try:
                self.load()
            except FileNotFoundError as e:
                print(f"Dataset '{self.name}' does not exist. You must create it")
                if require_exists:
                    raise e

    def get_element(self, unique_id):
        if unique_id in self.ind_for_train_id:
            ind = self.ind_for_train_id[unique_id]
            elem = self.train_md[ind]
            return self.meta_dataset_type([elem])
        if unique_id in self.ind_for_test_id:
            ind = self.ind_for_test_id[unique_id]
            elem = self.test_md[ind]
            return self.meta_dataset_type([elem])
        raise KeyError(f"Id {unique_id} not found in dataset")

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
        fp = self.get_save_path()
        if not fp.exists():
            raise FileNotFoundError(f"metadataset file not found: {fp.as_posix()}")
        with fp.open('rb') as f:
            self.__dict__ = pickle.load(f)

    @abc.abstractmethod
    def get_save_path(self):
        pass

    def get_training(self):
        return self.meta_dataset_type(self.train_md)

    def get_testing(self):
        return self.meta_dataset_type(self.test_md)


class ShapenetDatasetSupervisor(DatasetSupervisor):
    def __init__(self, name, require_exists=True, **kwargs):
        super().__init__(name, meta_dataset_type=ShapenetMetaDataset, require_exists=require_exists, **kwargs)

    def get_save_path(self):
        return get_shapenet_path() / "MetaDataSets" / f'{self.name}.metadataset.pkl'


def get_unique_name(datum, has_batch_dim=False):
    """
    Returns a unique name for the datum
    @param datum:
    @return:
    """
    # if has_batch_dim:
    #     return (datum['id']+ datum['augmentation'])
    return datum['id'] + datum['augmentation']


def get_all_shapenet_files(shape_ids):
    shapenet_records = []
    if shape_ids == "all":
        shape_ids = [f.name for f in get_shapenet_path().iterdir() if f.is_dir()]
        # shape_ids = [f for f in os.listdir(shapenet_load_path)
        #              if os.path.isdir(join(shapenet_load_path, f))]
        shape_ids.sort()

    # TODO: Add progressbar
    for category in shape_ids:
        shape_path = get_shapenet_path() / category
        for obj_fp in sorted(p for p in shape_path.iterdir()):
            # print("{}".format(obj_fp.name))
            all_augmentations = [f for f in (obj_fp / "models").iterdir()
                                 if f.name.startswith("model_augmented")
                                 if f.name.endswith(".pkl.gzip")]
            for f in sorted(all_augmentations):
                # shapenet_records.append(load_gt_voxels(f))
                base = f.parent / f.stem
                shapenet_records.append(load_metadata(base, compression="gzip"))

    return shapenet_records


def get_shapenet_path():
    return get_dataset_path('shapenet')

def get_dataset_path(dataset_name):
    config = get_config()
    dataset_path_name = f"{dataset_name}_path"
    if dataset_path_name not in config:
        raise KeyError(f"{dataset_path_name} must be defined in config.hjson file")
    p = Path(config[dataset_path_name])
    if p.is_absolute():
        return p
    return filepath_tools.get_shape_completion_package_path() / p


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
