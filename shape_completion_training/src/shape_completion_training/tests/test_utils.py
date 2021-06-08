from unittest import TestCase

import numpy as np
import tensorflow as tf
from shape_completion_training.utils import tf_utils
from shape_completion_training.voxelgrid.utils import inflate_voxelgrid


class TestUtils(TestCase):
    def test_geometric_mean(self):
        t = tf.convert_to_tensor([[1, 3, 9], [1, 1, 27.]])
        self.assertEqual(3, tf_utils.reduce_geometric_mean(t))

    def test_inflate_voxelgrid_by_one(self):
        vg_np = np.zeros((1, 13, 13, 13, 1), dtype=np.float32)
        vg_np[0, 7, 8, 11, 0] = 1.0
        vg = tf.constant(vg_np)
        inflated_vg = inflate_voxelgrid(vg)

        self.assertEqual(np.sum(inflated_vg), 27)

        for i in [6, 7, 8]:
            for j in [7, 8, 9]:
                for k in [10, 11, 12]:
                    self.assertEqual(inflated_vg[0, i, j, k, 0], 1.0)
