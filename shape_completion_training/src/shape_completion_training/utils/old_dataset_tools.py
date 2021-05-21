import pickle
import sys
from functools import lru_cache

import tensorflow as tf
from deprecated import deprecated

from shape_completion_training.model import filepath_tools
from shape_completion_training.utils import dataset_supervisor, tf_utils
from shape_completion_training.utils.data_tools import simulate_2_5D_input, select_slit_location, \
    simulate_slit_occlusion, simulate_2_5D_known_free, shift_dataset_element
from shape_completion_training.utils.dataset_storage import load_gt_only, _split_train_and_test, write_to_filelist
from shape_completion_training.utils.exploratory_data_tools import simulate_partial_completion, \
    simulate_random_partial_completion
from shape_completion_training.utils.dataset_supervisor import get_shapenet_path, get_all_shapenet_files
from shape_completion_training.utils.tf_utils import memoize
from shape_completion_training.voxelgrid import conversions

ycb_load_path = filepath_tools.get_shape_completion_package_path() / "data" / "ycb"
ycb_record_path = ycb_load_path / "tfrecords" / "filepath"


def get_dataset_path(dataset_name):
    paths = {"shapenet": dataset_supervisor.get_shapenet_path(),
             "ycb": ycb_load_path}
    return paths[dataset_name]


def load_dataset(dataset_name, metadata_only=True, shuffle=True):
    """
    @param shuffle: shuffle the dataset
    @param dataset_name: either "ycb" or "shapenet"
    @param metadata_only: if True, only loads metadata without voxelgrids
    """
    if dataset_name == 'shapenet':
        train_data, test_data = load_shapenet_metadata([
            dataset_supervisor.get_shapenet_map()["mug"]], shuffle=shuffle)
    elif dataset_name == 'ycb':
        train_data, test_data = load_ycb_metadata(shuffle=shuffle)
    else:
        raise Exception("Unknown dataset: {}".format(dataset_name))

    if not metadata_only:
        train_data = load_voxelgrids(train_data)
        test_data = load_voxelgrids(test_data)

    return train_data, test_data


def load_shapenet_metadata(shapes="all", shuffle=True):
    print("Loading Shapenet dataset")
    return _load_metadata_train_or_test(shapes, shuffle, "train"), \
           _load_metadata_train_or_test(shapes, shuffle, "test"),


def load_ycb_metadata(shuffle=True):
    print("Loading YCB dataset")
    return _load_metadata_train_or_test(shuffle=shuffle, prefix="train", record_path=ycb_record_path), \
           _load_metadata_train_or_test(shuffle=shuffle, prefix="test", record_path=ycb_record_path),


def preprocess_dataset(dataset, params):
    dataset = simulate_input(dataset,
                             params['translation_pixel_range_x'],
                             params['translation_pixel_range_y'],
                             params['translation_pixel_range_z'],
                             sim_input_fn=simulate_2_5D_input)

    if params['apply_depth_sensor_noise']:
        dataset = apply_sensor_noise(dataset)

    if params['apply_slit_occlusion']:
        print("Applying slit occlusion")
        dataset = apply_slit_occlusion(dataset)

    # Experimental processing used in exploratory methods. Not used in main paper
    if params['simulate_partial_completion']:
        dataset = simulate_partial_completion(dataset)
    if params['simulate_random_partial_completion']:
        dataset = simulate_random_partial_completion(dataset)
    return dataset


def preprocess_test_dataset(dataset, params):
    dataset = simulate_input(dataset, 0, 0, 0, sim_input_fn=simulate_2_5D_input)

    if params['apply_depth_sensor_noise']:
        dataset = apply_sensor_noise(dataset)

    if params['apply_slit_occlusion']:
        print("Applying fixed slit occlusion")
        dataset = apply_fixed_slit_occlusion(dataset, params['slit_start'], params['slit_width'])

    return dataset


@deprecated()
@lru_cache()
def get_shapenet_record_path():
    return get_shapenet_path() / "tfrecords" / "filepath"


def _load_metadata_train_or_test(shapes="all", shuffle=True, prefix="train",
                                 record_path=get_shapenet_record_path()):
    """
    Loads either the test or train data
    @param shapes: "all", or a list of shape names to load
    @param shuffle: True shuffles the dataset
    @param prefix: "train" or "test"
    @param record_path: pathlib path to the record files
    @return:
    """
    records = [f for f in record_path.iterdir()
               if f.name == prefix + "_filepaths.pkl"]
    if shapes != "all":
        print("Not yet handling partial loading")

    ds = None
    for fp in records:
        if ds:
            ds = ds.concatenate(read_metadata_from_filelist(fp, shuffle))
        else:
            ds = read_metadata_from_filelist(fp, shuffle)
    return ds


def read_metadata_from_filelist(record_file, shuffle):
    """
    Returns a tensorflow dataset from a record list of filepaths
    This dataset has the voxelgrid shape information, but not the actual voxelgrids,
    so that loading and iterating through the dataset is fast
    @param record_file:
    @param shuffle:
    @return:
    """

    # Note: Commented below is the "tensorflow" way of reading and writing a dataset.
    #  I got fed up with the boilerplate required to change fields and other random annoyances.
    #  Instead I just store .pkl files
    #
    # print("Reading from filepath record")
    # raw_dataset = tf.data.TFRecordDataset(record_file.as_posix())
    #
    # keys = ['id', 'shape_category', 'fp', 'augmentation']
    # tfrecord_description = {k: tf.io.FixedLenFeature([], tf.string) for k in keys}
    #
    # def _parse_record_function(example_proto):
    #     # Parse the input tf.Example proto using the dictionary above.
    #     example = tf.io.parse_single_example(example_proto, tfrecord_description)
    #     # return pickle.loads(example_proto.numpy())
    #     return example
    #
    # if shuffle:
    #     raw_dataset = raw_dataset.shuffle(10000)
    #
    # parsed_dataset = raw_dataset.map(_parse_record_function)
    is_python2 = sys.version_info < (3, 0)
    with open(record_file.as_posix(), 'rb') as f:
        if is_python2:
            filelist = pickle.load(f)
        else:
            filelist = pickle.load(f, encoding='latin1')
    ds = tf.data.Dataset.from_tensor_slices(filelist)

    if shuffle:
        ds = ds.shuffle(10000)

    return ds


def load_voxelgrids(metadata_ds):
    def _get_shape(_raw_dataset):
        e = next(_raw_dataset.__iter__())
        return tf.TensorShape(e["shape"])

    shape = _get_shape(metadata_ds)

    def _load_voxelgrids(elem):
        fp = elem['filepath']
        gt = tf.numpy_function(load_gt_only, [fp], tf.float32)
        gt.set_shape(shape)
        elem['gt_occ'] = gt
        elem['gt_free'] = 1.0 - gt

        return elem

    return metadata_ds.map(_load_voxelgrids)


def shift_bounding_box_only(elem, x, y, z):
    """
    Shift only the bounding box of elem by a random amount, up to the limits [x,y,z]
    :param elem:
    :param x: maximum x shift
    :param y: maximum y shift
    :param z: maximum z shift
    :return:
    """
    dx = 0
    dy = 0
    dz = 0
    if x > 0:
        dx = tf.random.uniform(shape=[], minval=-x, maxval=x, dtype=tf.int64)
    if y > 0:
        dy = tf.random.uniform(shape=[], minval=-y, maxval=y, dtype=tf.int64)
    if z > 0:
        dz = tf.random.uniform(shape=[], minval=-z, maxval=z, dtype=tf.int64)
    elem['bounding_box'] += tf.cast([[dx, dy, dz]], tf.float64) * 0.01
    return elem


def simulate_input(dataset, x, y, z, sim_input_fn=simulate_2_5D_input):
    def _simulate_input(example):
        known_occ, known_free = tf.numpy_function(sim_input_fn, [example['gt_occ']],
                                                  [tf.float32, tf.float32])
        known_occ.set_shape(example['gt_occ'].shape)
        known_free.set_shape(example['gt_occ'].shape)
        example['known_occ'] = known_occ
        example['known_free'] = known_free
        return example

    def _shift(elem):
        return shift_dataset_element(elem, x, y, z)

    return dataset.map(_shift, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(_simulate_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def apply_slit_occlusion(dataset):
    def _apply_slit_occlusion(elem):
        slit_min, slit_max = select_slit_location(elem['gt_occ'], min_slit_width=5, max_slit_width=30,
                                                  min_observable=5)
        ko, kf = tf.numpy_function(simulate_slit_occlusion, [elem['known_occ'], elem['known_free'],
                                                             slit_min, slit_max], [tf.float32, tf.float32])

        # ko, kf = simulate_slit_occlusion(elem['known_occ'].numpy(), elem_raw['known_free'].numpy(),
        #                              slitmin, slitmax)
        elem['known_occ'] = ko
        elem['known_free'] = kf
        return elem

    return dataset.map(_apply_slit_occlusion)


def helper_apply_sensor_noise(elem):
    img = conversions.to_2_5D(elem['known_occ'])
    noise = tf.random.normal(img.shape, stddev=1.0) * tf.cast(img < 64, tf.float32)
    img = img + noise

    ko = conversions.img_to_voxelgrid(img)
    elem['known_occ'] = ko
    return elem


def apply_sensor_noise(dataset):
    def _apply_sensor_noise(elem):
        img = conversions.to_2_5D(elem['known_occ'])
        noise = tf.random.normal(img.shape, stddev=1.0) * tf.cast(img < 64, tf.float32)
        img = img + noise

        ko = conversions.img_to_voxelgrid(img)
        elem['known_occ'] = ko
        elem['known_free'], = tf.numpy_function(simulate_2_5D_known_free, [ko],
                                                [tf.float32])
        return elem

    return dataset.map(_apply_sensor_noise)


def apply_fixed_slit_occlusion(dataset, slit_min, slit_width):
    def _apply_slit_occlusion(elem):
        ko, kf = tf.numpy_function(simulate_slit_occlusion, [elem['known_occ'], elem['known_free'],
                                                             slit_min, slit_min + slit_width], [tf.float32, tf.float32])
        elem['known_occ'] = ko
        elem['known_free'] = kf
        return elem

    return dataset.map(_apply_slit_occlusion)


def get_unique_name(datum, has_batch_dim=False):
    """
    Returns a unique name for the datum
    @param datum:
    @return:
    """
    if has_batch_dim:
        return (datum['id'].numpy() + datum['augmentation'].numpy())[0].decode('UTF-8')
    return (datum['id'].numpy() + datum['augmentation'].numpy()).decode('UTF-8')


@memoize
def get_addressible_dataset(**kwargs):
    return AddressableDataset(**kwargs)


class AddressableDataset:
    """
    Shape dataset where shapes can be looked up by name or index. Useful for
    Manually visualizing and examining shapes
    """

    def __init__(self, use_test=True, use_train=True, dataset_name="shapenet"):
        self.train_ds, self.test_ds = load_dataset(dataset_name=dataset_name,
                                                   metadata_only=True,
                                                   shuffle=False)
        self.train_map = {}
        self.test_map = {}
        self.train_names = []
        self.test_names = []

        if use_train:
            for i, elem in self.train_ds.enumerate():
                self.train_map[get_unique_name(elem)] = i
                self.train_names.append(get_unique_name(elem))
        if use_test:
            for i, elem in self.test_ds.enumerate():
                self.test_map[get_unique_name(elem)] = i
                self.test_names.append(get_unique_name(elem))

    def get(self, unique_name, params=None):
        if unique_name in self.train_map:
            ds = self.train_ds.skip(self.train_map[unique_name])
        elif unique_name in self.test_map:
            ds = self.test_ds.skip(self.test_map[unique_name])
        else:
            raise Exception("No element {} in dataset".format(unique_name))

        ds = load_voxelgrids(ds.take(1))
        if params is None:
            ds = simulate_input(ds, 0, 0, 0)
            return next(ds.__iter__())

        ds = preprocess_dataset(ds, params)
        return next(ds.__iter__())

    def get_metadata(self, unique_name):
        if unique_name in self.train_map:
            ds = self.train_ds.skip(self.train_map[unique_name])
        elif unique_name in self.test_map:
            ds = self.test_ds.skip(self.test_map[unique_name])
        else:
            raise Exception("No element {} in dataset".format(unique_name))
        return next(ds.__iter__())


@deprecated()
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