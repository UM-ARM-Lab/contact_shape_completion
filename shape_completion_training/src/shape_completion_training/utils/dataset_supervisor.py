import abc
import pickle
from functools import lru_cache
from pathlib import Path

import hjson
import progressbar
from colorama import Fore

from shape_completion_training.model import filepath_tools
from shape_completion_training.utils import data_tools
from shape_completion_training.utils.config import get_config
from shape_completion_training.utils.data_tools import simulate_2_5D_input
from shape_completion_training.utils.dataset_storage import load_metadata, _split_train_and_test, load_gt_only
from shape_completion_training.utils.tf_utils import sequence_of_dicts_to_dict_of_sequences, stack_dict

"""
Tools for storing and preprocessing augmented shapenet
"""


class MetaDataset(abc.ABC):
    load_limit = 20

    def __init__(self, metadata, params):
        self.md = metadata
        self.params = params

    @abc.abstractmethod
    def absolute_fp(self, rel_fp):
        pass

    def batch(self, batch_size):
        for i in range(0, len(self.md), batch_size):
            yield self.__class__(self.md[i:i + batch_size], self.params)

    def size(self):
        return len(self.md)

    def load(self):
        if "load_bb_only" in self.params and self.params['load_bb_only']:
            return self.load_bounding_box_only()
        return self.load_all()

    def load_bounding_box_only(self):
        for elem in self.md:
            x, y, z = (self.params[f'translation_pixel_range_{axis}'] for axis in ['x', 'y', 'z'])
            elem['bounding_box'] = data_tools.shift_bounding_box_only(elem['bounding_box'], x, y, z).numpy()
        return stack_dict(sequence_of_dicts_to_dict_of_sequences(self.md))

    def load_all(self):
        if len(self.md) > self.load_limit:
            raise OverflowError(f"Asked to load {len(self.md)} shapes. Too large. You probably didnt mean that")
        for elem in self.md:
            rel_fp = elem['filepath']
            fp = self.absolute_fp(rel_fp)
            vg = load_gt_only(fp)
            elem['gt_occ'] = vg
            elem['gt_free'] = 1 - vg
            x, y, z = (self.params[f'translation_pixel_range_{axis}'] for axis in ['x', 'y', 'z'])
            data_tools.shift_dataset_element(elem, x, y, z)
            elem['gt_occ'] = elem['gt_occ'].numpy()
            elem['gt_free'] = elem['gt_free'].numpy()
            elem['bounding_box'] = elem['bounding_box'].numpy()
            ko, kf = simulate_2_5D_input(elem['gt_occ'])

            if self.params[f'apply_slit_occlusion']:
                slit_min, slit_max = data_tools.select_slit_location(elem['gt_occ'], min_slit_width=5,
                                                                     max_slit_width=30, min_observable=5)
                ko, kf = data_tools.simulate_slit_occlusion(ko, kf, slit_min, slit_max)

            elem['known_occ'] = ko
            elem['known_free'] = kf
        return stack_dict(sequence_of_dicts_to_dict_of_sequences(self.md))


class ShapenetMetaDataset(MetaDataset):
    def __init__(self, metadata, params):
        super().__init__(metadata, params)

    def absolute_fp(self, rel_fp):
        # TODO: This fixes a temporary bug with the way the filepath is saved if the shapenet path is relative,
        #  (not absolute)
        prefix = "data/ShapeNetCore.v2_augmented/"
        if rel_fp.startswith(prefix):
            rel_fp = rel_fp[len(prefix):]
        fp = get_shapenet_path() / rel_fp
        return fp


class YcbMetaDataset(MetaDataset):
    def __init__(self, metadata, params):
        super().__init__(metadata, params)

    def absolute_fp(self, rel_fp):
        # Temporary fix for incorrect storage of path
        prefix = "data/ycb/"
        if rel_fp.startswith(prefix):
            rel_fp = rel_fp[len(prefix):]
        return get_dataset_path('ycb') / rel_fp


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

    def get_element(self, unique_id, params):
        if unique_id in self.ind_for_train_id:
            ind = self.ind_for_train_id[unique_id]
            elem = self.train_md[ind]
            return self.meta_dataset_type([elem], params)
        if unique_id in self.ind_for_test_id:
            ind = self.ind_for_test_id[unique_id]
            elem = self.test_md[ind]
            return self.meta_dataset_type([elem], params)
        raise KeyError(f"Id {unique_id} not found in dataset")

    @abc.abstractmethod
    def create_new_dataset(self, shape_ids, test_ratio=0.1):
        pass

    def save(self, overwrite=False):
        fp = self.get_save_path()
        # fp = get_shapenet_path() / "MetaDataSets"
        # fp.mkdir(exist_ok=True)
        # fp = fp / f'{self.name}.metadataset.pkl'
        fp.parent.mkdir(exist_ok=True)
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

    def get_training(self, params):
        return self.meta_dataset_type(self.train_md, params)

    def get_testing(self, params):
        return self.meta_dataset_type(self.test_md, params)


class ShapenetDatasetSupervisor(DatasetSupervisor):
    def __init__(self, name, require_exists=True, **kwargs):
        super().__init__(name, meta_dataset_type=ShapenetMetaDataset, require_exists=require_exists, **kwargs)

    def get_save_path(self):
        return get_shapenet_path() / "MetaDataSets" / f'{self.name}.metadataset.pkl'

    def create_new_dataset(self, shape_ids, test_ratio=0.1):
        files = get_all_shapenet_files(shape_ids)
        train_files, test_files = _split_train_and_test(files, test_ratio)
        self.train_md = train_files
        self.test_md = test_files

        for i, elem in enumerate(self.train_md):
            self.ind_for_train_id[get_unique_name(elem)] = i
        for i, elem in enumerate(self.test_md):
            self.ind_for_test_id[get_unique_name(elem)] = i


class YcbDatasetSupervisor(DatasetSupervisor):
    def __init__(self, name, require_exists=True, **kwargs):
        super().__init__(name, meta_dataset_type=YcbMetaDataset, require_exists=require_exists, **kwargs)

    def get_save_path(self):
        return get_dataset_path('ycb') / "MetaDataSets" / f'{self.name}.metadataset.pkl'

    def create_new_dataset(self, shape_ids, test_ratio=0.1):
        files = get_all_ycb_files(shape_ids)
        train_files, test_files = _split_train_and_test(files, test_ratio)
        self.train_md = train_files
        self.test_md = test_files

        for i, elem in enumerate(self.train_md):
            self.ind_for_train_id[get_unique_name(elem)] = i
        for i, elem in enumerate(self.test_md):
            self.ind_for_test_id[get_unique_name(elem)] = i


def get_dataset_supervisor(dataset: str):
    if dataset.startswith("shapenet"):
        print(f"{Fore.GREEN}Loading Shapenet Dataset {dataset}{Fore.RESET}")
        return ShapenetDatasetSupervisor(dataset)
    elif dataset.startswith("ycb"):
        print(f"{Fore.GREEN}Loading YCB Dataset {dataset}{Fore.RESET}")
        return YcbDatasetSupervisor(dataset)
    raise RuntimeError(f"Error: Unknown dataset {dataset}")


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
    if shape_ids == "all":
        shape_ids = [f.name for f in get_shapenet_path().iterdir() if f.is_dir()]
        # shape_ids = [f for f in os.listdir(shapenet_load_path)
        #              if os.path.isdir(join(shapenet_load_path, f))]
        shape_ids.sort()

    files = []

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
                files.append(base)
    shapenet_records = []
    for file in progressbar.progressbar(files):
        shapenet_records.append(load_metadata(file, compression="gzip"))
    return shapenet_records


def get_all_ycb_files(shape_ids):
    records = []
    # obj_fps = [fp for fp in get_dataset_path('ycb').iterdir() if fp.stem.startswith("0")]
    for category in shape_ids:
        # for obj_fp in sorted(obj_fps):
        obj_fp = get_dataset_path('ycb') / category

        print("\t{}".format(obj_fp.name))
        all_augmentation = [f for f in (obj_fp / "google_16k").iterdir()
                            if f.name.startswith("model_augmented")
                            if f.name.endswith(".pkl.gzip")]
        for f in sorted(all_augmentation):
            base = f.parent / f.stem
            records.append(load_metadata(base, compression="gzip"))
    return records


def get_shapenet_path():
    return get_dataset_path('shapenet')


@lru_cache()
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
def get_shapenet_map():
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
    return [get_shapenet_map()[hn] for hn in human_names]
