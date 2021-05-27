import numpy as np
from tensorflow import TensorShape

from shape_completion_training.utils import data_tools
from shape_completion_training.utils.data_tools import simulate_2_5D_input
from shape_completion_training.utils.dataset_supervisor import DatasetSupervisor, MetaDataset
from shape_completion_training.utils.tf_utils import stack_dict, sequence_of_dicts_to_dict_of_sequences

VOXELGRID_SIZE = 64


class AabMetaDataset(MetaDataset):
    def __init__(self, metadata, params):
        super().__init__(metadata, params)

    def load_bounding_box_only(self):
        for elem in self.md:
            x, y, z = (self.params[f'translation_pixel_range_{axis}'] for axis in ['x', 'y', 'z'])
            elem['bounding_box'] = data_tools.shift_bounding_box_only(elem['bounding_box'], x, y, z).numpy()
        return stack_dict(sequence_of_dicts_to_dict_of_sequences(self.md))

    def load_all(self):
        for elem in self.md:
            x, y, z = (self.params[f'translation_pixel_range_{axis}'] for axis in ['x', 'y', 'z'])
            elem['bounding_box'] = data_tools.shift_bounding_box_only(elem['bounding_box'], x, y, z).numpy()
            ind_corners = elem['bounding_box'] / elem['scale']
            mins = np.clip(np.min(ind_corners, axis=0).astype(int), 0, VOXELGRID_SIZE - 1)
            maxs = np.clip(np.max(ind_corners, axis=0).astype(int), 0, VOXELGRID_SIZE - 1)
            gt = np.zeros(elem['shape'])
            gt[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]] = 1.0

            if np.sum(gt) == 0:
                raise RuntimeError("Empty Voxel Grid")

            elem['gt_occ'] = gt
            elem['gt_free'] = 1 - elem['gt_occ']
            ko, kf = simulate_2_5D_input(elem['gt_occ'])

            if self.params[f'apply_slit_occlusion']:
                slit_min, slit_max = data_tools.select_slit_location(elem['gt_occ'], min_slit_width=5,
                                                                     max_slit_width=70, min_observable=5)
                ko, kf = data_tools.simulate_slit_occlusion(ko, kf, slit_min, slit_max)

                if np.sum(kf) == 0:
                    raise RuntimeError("Empty Known Free")

            elem['known_occ'] = ko
            elem['known_free'] = kf

        return stack_dict(sequence_of_dicts_to_dict_of_sequences(self.md))

    def absolute_fp(self, rel_fp):
        raise RuntimeError("absolute fp is not needed for aab dataset")


class AabDatasetSupervisor(DatasetSupervisor):
    def __init__(self, name, require_exists=True, length=10000, **kwargs):
        kwargs['load'] = False
        super().__init__(name, meta_dataset_type=AabMetaDataset, require_exists=require_exists, **kwargs)

        self.length = length
        self.train_md = [self._generate_random_box_md() for _ in range(length)]
        self.test_md = [self._generate_random_box_md() for _ in range(length)]

        for i, elem in enumerate(self.train_md):
            self.ind_for_train_id[f"{elem['id']}_{elem['augmentation']}"] = i
        for i, elem in enumerate(self.test_md):
            self.ind_for_test_id[f"{elem['id']}_{elem['augmentation']}"] = i

    @staticmethod
    def _generate_random_box_md():
        w = np.random.randint(2, 41)
        h = np.random.randint(2, 41)
        l = np.random.randint(2, 41)
        scale = 0.01

        bb = (np.array([[x, y, z] for x in [-w/2, w/2] for y in [-l/2, l/2] for z in [-h/2, h/2]]) + 32) * scale
        # bb += 32 - np.array([w/2])

        d = {'shape': TensorShape((64, 64, 64, 1)),
             'augmentation': f'{l}_{w}_{h}',
             'w': w,
             'h': h,
             'l': l,
             'y_angle': 0,
             'x_angle': 0,
             'z_angle': 0,
             'category': 'Axis Aligned Box',
             'id': 'Axis Aligned Box',
             'scale': scale,
             'bounding_box': bb,
             }
        return d

    def get_save_path(self):
        raise RuntimeError("There is no save path for AabDatasets")

    def create_new_dataset(self, shape_ids, test_ratio=0.1):
        raise RuntimeError("There is no need to create a new AabDataset")
