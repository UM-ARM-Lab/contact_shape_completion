import unittest

from contact_shape_completion import contact_tools
import tensorflow as tf


def expand(t):
    return tf.expand_dims(tf.expand_dims(t, axis=-1), axis=0)


def create_tensor(t):
    return expand(tf.convert_to_tensor(t))


class TestPSSNet(unittest.TestCase):
    def test_get_assumed_occ(self):
        pred_occ = create_tensor([[[0.1, 0.2], [0.3, 0.4]],
                                  [[0.2, 0.2], [0.21, 0.2]]])

        chs_a = create_tensor([[[1.0, 1.0], [1.0, 1.0]],
                               [[0.0, 0.0], [0.0, 0.0]]])

        chs_b = create_tensor([[[0.0, 0.0], [0.0, 0.0]],
                               [[1.0, 1.0], [1.0, 1.0]]])

        expected_occ = create_tensor([[[0.0, 0.0], [0.0, 1.0]],
                                      [[0.0, 0.0], [1.0, 0.0]]])

        chss = tf.concat([chs_a, chs_b], axis=0)

        res = contact_tools.get_assumed_occ(pred_occ, chss)

        self.assertEqual(res.shape[0], 1)
        self.assertTrue(tf.reduce_all(res == expected_occ))

    def test_get_assumed_occ_with_no_chss(self):
        pred_occ = create_tensor([[[0.1, 0.2], [0.3, 0.4]],
                                  [[0.2, 0.2], [0.21, 0.2]]])
        chss = None
        expected_occ = create_tensor([[[0.0, 0.0], [0.0, 0.0]],
                                      [[0.0, 0.0], [0.0, 0.0]]])
        res = contact_tools.get_assumed_occ(pred_occ, chss)
        self.assertEqual(res.shape[0], 1)
        self.assertTrue(tf.reduce_all(res == expected_occ))


if __name__ == '__main__':
    unittest.main()
