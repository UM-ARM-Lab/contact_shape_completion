#! /usr/bin/env python
from __future__ import print_function

import tensorflow as tf

from shape_completion_training.voxelgrid import conversions
import numpy as np


# TODO: Hardcoded param for converting pixels into meters
PIXELS_TO_METERS = 0.01

def simulate_2_5D_input(gt):
    gt_occ = gt
    gt_free = 1.0 - gt
    known_occ = gt + 0.0
    known_free = gt_free + 0.0
    unknown_mask = np.zeros((gt.shape[1], gt.shape[2]))
    for h in range(gt.shape[0]):
        known_occ[h, :, :, 0] = np.clip(known_occ[h, :, :, 0] - unknown_mask, 0, 1)
        known_free[h, :, :, 0] = np.clip(known_free[h, :, :, 0] - unknown_mask, 0, 1)
        unknown_mask = unknown_mask + gt_occ[h, :, :, 0]
        unknown_mask = np.clip(unknown_mask, 0, 1)
    return known_occ, known_free


def simulate_2_5D_known_free(known_occ):
    unknown_mask = np.zeros((known_occ.shape[1], known_occ.shape[2]))
    known_free = 1.0 - known_occ
    for h in range(known_occ.shape[0]):
        known_free[h, :, :, 0] = np.clip(known_free[h, :, :, 0] - unknown_mask, 0, 1)
        unknown_mask = np.clip(unknown_mask + known_occ[h, :, :, 0], 0, 1)
    return known_free


def simulate_slit_occlusion(known_occ, known_free, slit_zmin, slit_zmax):
    known_occ[:, :, 0:slit_zmin, 0] = 0
    known_free[:, :, 0:slit_zmin, 0] = 0

    known_occ[:, :, slit_zmax:, 0] = 0
    known_free[:, :, slit_zmax:, 0] = 0
    return known_occ, known_free


def get_slit_occlusion_2D_mask(slit_min, slit_width, mask_shape):
    slit_max = slit_min + slit_width
    mask = np.zeros(mask_shape)
    mask[:, 0:slit_min] = 1.0
    mask[:, slit_max:] = 1.0
    return mask


def select_slit_location(gt, min_slit_width, max_slit_width, min_observable=5):
    """
    Randomly select a slit location
    @param gt: voxelgrid of shape
    @param min_slit_width:
    @param max_slit_width:
    @param min_observable: minimum columns of voxelgrid that must be visible
    @return:
    """
    z_vals = tf.where(tf.reduce_sum(gt, axis=[0, 1, 3]))

    slit_width = tf.random.uniform(shape=[], minval=min_slit_width, maxval=max_slit_width, dtype=tf.int64)

    slit_min_possible = tf.reduce_min(z_vals) - slit_width + min_observable
    slit_max_possible = tf.reduce_max(z_vals) - min_observable
    slit_max_possible = tf.maximum(slit_max_possible, slit_min_possible + 1)

    slit_min = tf.random.uniform(shape=[],
                                 minval=slit_min_possible,
                                 maxval=slit_max_possible,
                                 dtype=tf.int64)

    return tf.clip_by_value(slit_min, 0, 63), tf.clip_by_value(slit_min + slit_width, 0, 63)


def simulate_depth_image(vg):
    """Note: I have a more efficient way to do this now. See conversions.py"""
    vg = conversions.format_voxelgrid(vg, False, False)
    size = vg.shape[1]
    z_inds = tf.expand_dims(tf.expand_dims(tf.range(size), axis=-1), axis=-1)
    z_inds = tf.repeat(tf.repeat(z_inds, size, axis=1), size, axis=2)
    z_inds = tf.cast(z_inds, tf.float32)
    dists = z_inds * vg + size * tf.cast(vg == 0, tf.float32)
    return tf.reduce_min(dists, axis=0)


@tf.function
def shift_voxelgrid(t, dx, dy, dz, pad_value, max_x, max_y, max_z):
    """
    Shifts a single (non-batched) voxelgrid of shape (x,y,z,channels)
    :param t: voxelgrid tensor to shift
    :param dx: x shift amount (tensor of shape [])
    :param dy: y shift amount
    :param dz: z shift amount
    :param pad_value: value to pad the new "empty" spaces
    :param max_x: max x shift
    :param max_y: max y shift
    :param max_z: max z shift
    :return:
    """
    a = np.abs(max_x)
    b = np.abs(max_y)
    c = np.abs(max_z)

    if a > 0:
        t = tf.pad(t, paddings=tf.constant([[a, a], [0, 0], [0, 0], [0, 0]]),
                   mode="CONSTANT", constant_values=pad_value)
        t = tf.roll(t, [dx], axis=[0])
        t = t[a:-a, :, :, :]

    if b > 0:
        t = tf.pad(t, paddings=tf.constant([[0, 0], [b, b], [0, 0], [0, 0]]),
                   mode="CONSTANT", constant_values=pad_value)
        t = tf.roll(t, [dy], axis=[1])
        t = t[:, b:-b, :, :]
    if c > 0:
        t = tf.pad(t, paddings=tf.constant([[0, 0], [0, 0], [c, c], [0, 0]]),
                   mode="CONSTANT", constant_values=pad_value)
        t = tf.roll(t, [dz], axis=[2])
        t = t[:, :, c:-c, :]

    return t


if __name__ == "__main__":
    print("Not meant to be executed as main script")


def shift_dataset_element(elem, x, y, z):
    """
    Shift the voxelgrid and bounding box of elem by a random amount, up to the limits [x,y,z]
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
    elem['gt_occ'] = shift_voxelgrid(elem['gt_occ'], dx, dy, dz, 0.0, x, y, z)
    elem['gt_free'] = shift_voxelgrid(elem['gt_free'], dx, dy, dz, 1.0, x, y, z)
    elem['bounding_box'] += tf.cast([[dx, dy, dz]], tf.float64) * PIXELS_TO_METERS
    return elem


def shift_bounding_box_only(bounding_box, x, y, z):
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
    bounding_box += tf.cast([dx, dy, dz], tf.float64) * PIXELS_TO_METERS
    return bounding_box


def apply_sensor_noise(known_occ):
    img = conversions.to_2_5D(known_occ)
    noise_width = 4
    noise_shape = (np.array([img.shape[0], img.shape[1], noise_width]) / noise_width).astype(int)
    noise = tf.random.normal(noise_shape, stddev=2.0)
    noise = tf.image.resize(noise, img.shape[0:2])
    noise = noise * tf.cast(img < 64, tf.float32)
    img = img + noise

    ko = conversions.img_to_voxelgrid(img).numpy()

    return ko, simulate_2_5D_known_free(ko)
