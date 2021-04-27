import unittest

import rospy

import numpy as np
from pathlib import Path

from shape_completion_training.model.model_runner import ModelRunner
from contact_shape_completion.contact_shape_completer import ContactShapeCompleter
from contact_shape_completion.goal_generator import CheezeitGoalGenerator
from shape_completion_training.utils.tf_utils import add_batch_to_dict


class TestPSSNet(unittest.TestCase):
    def setUp(self):
        # rospy.init_node("test_pssnet")
        trial = 'PSSNet_YCB/July_24_11-21-46_f2aea4d768'
        self.pssnet = self.model_runner = ModelRunner(training=False, trial_path=trial).model
        # goal_generator = CheezeitGoalGenerator()
        # self.contact_shape_completer = contact_shape_completer = ContactShapeCompleter(trial, goal_generator=goal_generator)
        # self.contact_shape_completer.load_last_visible_vg()

    def test_enforce_freespace_results_in_completion_with_freespace(self):
        # self.contact_shape_completer.reset_completer_srv()
        # visible_vg = self.contact_shape_completer.last_visible_vg
        visible_vg = np.load((Path('.') / 'files/visible_vg.npz').as_posix())
        visible_vg = {k: v for k, v in visible_vg.items()}
        visible_vg = add_batch_to_dict(visible_vg)
        self.assertEqual(np.min(visible_vg['known_occ']), 0)
        self.assertEqual(np.max(visible_vg['known_occ']), 1)
        self.assertEqual(np.min(visible_vg['known_free']), 0)
        self.assertEqual(np.max(visible_vg['known_free']), 1)

        inference = self.pssnet(visible_vg)



if __name__ == '__main__':
    unittest.main()
