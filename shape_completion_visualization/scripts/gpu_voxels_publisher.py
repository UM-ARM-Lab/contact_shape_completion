#!/usr/bin/env python
from __future__ import print_function

import argparse
import math
import random
import time

import rospy
import numpy as np

import shape_completion_training.utils.old_dataset_tools
from rviz_voxelgrid_visuals.conversions import pointcloud2_msg_to_vox
from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.model import default_params
from shape_completion_training.utils import data_tools
from shape_completion_training.model.pssnet import PSSNet
from shape_completion_training.voxelgrid import metrics
from shape_completion_training.model.other_model_architectures import sampling_tools
# from shape_completion_training.voxelgrid import conversions
from rviz_voxelgrid_visuals.conversions import pointcloud2_msg_to_vox
from shape_completion_training.voxelgrid.metrics import chamfer_distance
from shape_completion_visualization.voxelgrid_publisher import VoxelgridPublisher, PointcloudPublisher
from shape_completion_visualization.shape_selection import send_display_names_from_metadata
from gpu_voxel_planning_msgs.srv import CompleteShape, CompleteShapeResponse, CompleteShapeRequest
from shape_completion_training.utils.tf_utils import log_normal_pdf, stack_known, sample_gaussian

import tensorflow as tf

"""
Publish object pointclouds for use in gpu_voxels planning
"""

ARGS = None
VG_PUB = None
PT_PUB = None

model_runner = None
dataset_params = None

default_dataset_params = default_params.get_default_params()

default_translations = {
    'translation_pixel_range_x': 0,
    'translation_pixel_range_y': 0,
    'translation_pixel_range_z': 0,
}

Transformer = None


def wip_enforce_contact(elem):
    inference = model_runner.model(elem)
    VG_PUB.publish_inference(inference)
    pssnet = model_runner.model
    latent = tf.Variable(pssnet.sample_latent(elem))
    VG_PUB.publish('predicted_occ', pssnet.decode(latent, apply_sigmoid=True))
    known_contact = tf.Variable(tf.zeros((1, 64, 64, 64, 1)))
    known_free = tf.Variable(tf.zeros((1, 64, 64, 64, 1)))
    known_contact = known_contact[0, 50, 32, 32, 0].assign(1)
    VG_PUB.publish('aux', known_contact)

    rospy.sleep(2)

    for i in range(100):
        pssnet.grad_step_towards_output(latent, known_contact, known_free)
        VG_PUB.publish('predicted_occ', pssnet.decode(latent, apply_sigmoid=True))
    return pssnet.decode(latent, apply_sigmoid=True)


def sample_random_shape():
    print("Sampling Random Shapes")
    pssnet = model_runner.model  # type: PSSNet
    elem = get_elem(test_records, 90)
    for i in range(100):
        latent = tf.Variable(
            pssnet.apply_flow_to_latent_box(tf.random.normal(shape=[1, pssnet.hparams["num_latent_layers"]])))
        # latent = tf.Variable(pssnet.sample_latent(elem))

        vg = pssnet.decode(latent, apply_sigmoid=True)
        VG_PUB.publish('predicted_occ', vg)
        # PT_PUB.publish('predicted_occ', vg)
        rospy.sleep(0.1)


def complete_shape(req: CompleteShapeRequest):
    print("Complete Shape Request Received")
    pssnet = model_runner.model  # type: PSSNet
    elem = get_elem(test_records, 90)

    # Sample random latent tables
    # for i in range(100):
    #     # latent = tf.Variable(tf.random.normal(shape=[1, pssnet.hparams["num_latent_layers"]]))
    #     latent = tf.Variable(pssnet.sample_latent(elem))
    #     vg = pssnet.decode(latent, apply_sigmoid=True)
    #     VG_PUB.publish('predicted_occ', vg)
    #     # PT_PUB.publish('predicted_occ', vg)
    #     rospy.sleep(0.1)

    # latent = tf.Variable(tf.random.normal(shape=[1, pssnet.hparams["num_latent_layers"]]))
    latent = tf.Variable(pssnet.sample_latent(elem))
    mean, logvar = pssnet.encode(stack_known(elem))
    latent = tf.Variable(pssnet.apply_flow_to_latent_box(sample_gaussian(mean, logvar)))
    log_normal_pdf(latent, mean, logvar)

    p = lambda v: np.exp(log_normal_pdf(v, mean, logvar).numpy()[0])

    known_contact = tf.Variable(tf.zeros((1, 64, 64, 64, 1)))
    # known_free = tf.Variable(tf.zeros((1, 64, 64, 64, 1)))
    transformed_free = PT_PUB.transformer.transform_to_frame(req.known_free, "object")
    known_free = pointcloud2_msg_to_vox(transformed_free, scale=0.05)
    # VG_PUB.publish('known_free', known_free)
    # time.sleep(3)
    for i in range(10):
        vg = pssnet.decode(latent, apply_sigmoid=True)
        VG_PUB.publish('predicted_occ', vg)
        VG_PUB.publish('known_free', known_free)
        if i == 0:
            rospy.sleep(3)
        # PT_PUB.publish('predicted_occ', vg)
        pssnet.grad_step_towards_output(latent, known_contact, known_free)
        rospy.sleep(0.1)
    return CompleteShapeResponse()


def run_inference(elem):
    if ARGS.enforce_contact:
        return wip_enforce_contact(elem)

    if ARGS.publish_closest_train:
        # Computes and publishes the closest element in the training set to the test shape
        train_in_correct_augmentation = train_records.filter(lambda x: x['augmentation'] == elem['augmentation'][0])
        train_in_correct_augmentation = shape_completion_training.utils.old_dataset_tools.load_voxelgrids(train_in_correct_augmentation)
        min_cd = np.inf
        closest_train = None
        for train_elem in train_in_correct_augmentation:
            VG_PUB.publish("plausible", train_elem['gt_occ'])
            cd = chamfer_distance(elem['gt_occ'], train_elem['gt_occ'],
                                  scale=0.01, downsample=4)
            if cd < min_cd:
                min_cd = cd
                closest_train = train_elem['gt_occ']
            VG_PUB.publish("plausible", closest_train)

    # raw_input("Ready to display best?")

    inference = model_runner.model(elem)
    VG_PUB.publish_inference(inference)

    return inference


def get_elem(metadata, ind):
    ds = metadata.skip(ind).take(1)
    ds = shape_completion_training.utils.old_dataset_tools.load_voxelgrids(ds)
    ds = shape_completion_training.utils.old_dataset_tools.preprocess_test_dataset(ds, dataset_params)

    elem_raw = next(ds.__iter__())
    for k in elem_raw.keys():
        elem_raw[k] = tf.expand_dims(elem_raw[k], axis=0)
    return elem_raw


def publish_selection(metadata, ind, str_msg):
    if ind == 0:
        print("Skipping first display")
        return

    # translation = 0

    # ds = metadata.skip(ind).take(1)
    # ds = data_tools.load_voxelgrids(ds)
    # ds = data_tools.preprocess_test_dataset(ds, dataset_params)
    #
    # elem_raw = next(ds.__iter__())
    # elem = {}
    #
    # for k in elem_raw.keys():
    #     elem_raw[k] = tf.expand_dims(elem_raw[k], axis=0)
    elem_raw = get_elem(metadata, ind)
    elem = {}
    for k in elem_raw.keys():
        elem[k] = elem_raw[k].numpy()
    VG_PUB.publish_elem(elem)
    PT_PUB.publish('gt', elem['gt_occ'])

    if model_runner is None:
        return

    elem = sampling_tools.prepare_for_sampling(elem)

    inference = run_inference(elem_raw)
    if ARGS.enforce_contact:
        return

    # VG_PUB.publish_inference(inference)

    # fit_to_particles(metadata)

    mismatch = np.abs(elem['gt_occ'] - inference['predicted_occ'].numpy())
    VG_PUB.publish("mismatch", mismatch)
    # mismatch_pub.publish(to_msg(mismatch))
    print("There are {} mismatches".format(np.sum(mismatch > 0.5)))


def load_network():
    global model_runner
    # global model_evaluator
    if ARGS.trial is None:
        print("Not loading any inference model")
        return
    model_runner = ModelRunner(training=False, trial_path=ARGS.trial)
    # model_evaluator = ModelEvaluator(model_runner.model)


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
    parser.add_argument('--sample', help='foo help', action='store_true')
    parser.add_argument('--use_best_iou', help='foo help', action='store_true')
    parser.add_argument('--publish_each_sample', help='foo help', action='store_true')
    parser.add_argument('--fit_to_particles', help='foo help', action='store_true')
    parser.add_argument('--publish_nearest_plausible', help='foo help', action='store_true')
    parser.add_argument('--publish_nearest_sample', help='foo help', action='store_true')
    parser.add_argument('--multistep', action='store_true')
    parser.add_argument('--trial')
    parser.add_argument('--publish_closest_train', action='store_true')
    parser.add_argument('--enforce_contact', action='store_true')

    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_command_line_args()

    rospy.init_node('contact_shape_completer_service')
    rospy.loginfo("Data Publisher")

    load_network()

    dataset_params = default_dataset_params
    if model_runner is not None:
        dataset_params.update(model_runner.params)
        dataset_params.update({
            "slit_start": 32,
            "slit_width": 32,
        })
    # dataset_params.update({
    #     "apply_depth_sensor_noise": True,
    # })

    dataset_params.update(default_translations)
    # train_records, test_records = data_tools.load_dataset(dataset_name=dataset_params['dataset'],
    #                                                       metadata_only=True, shuffle=False)

    VG_PUB = VoxelgridPublisher(scale=0.05)
    PT_PUB = PointcloudPublisher(scale=0.05)
    # COMPLETE_SHAPE_SRV = rospy.Service("complete_shape", CompleteShape, complete_shape)

    # selection_sub = send_display_names_from_metadata(train_records, publish_selection)
    # selection_sub = send_display_names_from_metadata(test_records, publish_selection)

    # complete_shape(None)
    # sample_random_shape()
    print("Up and running")
    rospy.spin()
