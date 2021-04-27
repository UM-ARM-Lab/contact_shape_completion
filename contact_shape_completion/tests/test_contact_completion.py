#!/usr/bin/env python

import rospy

from contact_shape_completion.contact_shape_completer import ContactShapeCompleter
from contact_shape_completion.goal_generator import CheezeitGoalGenerator
from shape_completion_training.utils.tf_utils import add_batch_to_dict
import tensorflow as tf
import numpy as np


def assert_true(cond: bool, msg: str = None):
    if not cond:
        raise RuntimeError(f"Failed: {msg}")


def test_enforce_contact(sc):
    tf.random.set_seed(42)
    latent = tf.Variable(sc.model_runner.model.sample_latent(add_batch_to_dict(sc.last_visible_vg)))
    known_free = tf.Variable(tf.zeros((1, 64, 64, 64, 1)))
    chss = tf.Variable(tf.zeros((1, 64, 64, 64, 1)))

    initial_completion = sc.model_runner.model.decode(latent, apply_sigmoid=True)

    occ_inds = tf.where(initial_completion > 0.5)

    free_x = np.max(occ_inds.numpy()[:, 1]) - 3
    known_free_np = np.zeros(initial_completion.shape)
    known_free_np[:, free_x:, :, :, :] = 1.0
    known_free = tf.convert_to_tensor(known_free_np, tf.dtypes.float32)
    sc.robot_view.VG_PUB.publish("known_free", known_free)

    assert_true(tf.reduce_sum(known_free * initial_completion) > 1.0,
                "Known free does not overlap with initial completion. Test is useless")

    new_latent = sc.enforce_contact(latent, known_free, chss)

    enforced_completion = sc.model_runner.model.decode(new_latent, apply_sigmoid=True)

    assert_true(tf.reduce_max(known_free * enforced_completion) <= 0.5,
                "enforce_contact did not eliminate all known_free voxels")


if __name__ == "__main__":
    rospy.init_node('contact_shape_completer_service')
    rospy.loginfo("Data Publisher")

    trial = 'PSSNet_YCB/July_24_11-21-46_f2aea4d768'
    goal_generator = CheezeitGoalGenerator()
    contact_shape_completer = ContactShapeCompleter(trial, goal_generator=goal_generator)

    rospy.sleep(1)

    contact_shape_completer.get_visible_vg()
    contact_shape_completer.load_last_visible_vg()
    # contact_shape_completer.get_visible_vg()

    test_enforce_contact(contact_shape_completer)

    print("Up and running")
    # rospy.spin()
