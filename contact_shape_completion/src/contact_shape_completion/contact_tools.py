import dataclasses

import numpy as np
import tensorflow as tf
from colorama import Fore

import ros_numpy
from contact_shape_completion.beliefs import ParticleBelief
from contact_shape_completion.hardcoded_params import KNOWN_OCC_LIMIT, KNOWN_FREE_LIMIT, GRADIENT_UPDATE_ITERATION_LIMIT
from shape_completion_training.model.pssnet import PSSNet
from shape_completion_training.utils.tf_utils import log_normal_pdf
from shape_completion_training.voxelgrid import conversions
from shape_completion_visualization.voxelgrid_publisher import VoxelgridPublisher


def get_assumed_occ(pred_occ, chss):
    """
    Computes a matrix of "known_occupied", assuming each chs is satisfied by the most likely voxel from the predicted
    occupied
    Args:
        pred_occ: the predicted occupancy <1, 64, 64, 64, 1>
        chss: the chss <n, 64, 64, 64, 1>
    Returns:
    """
    if chss is None:
        return tf.zeros(pred_occ.shape)

    a = chss * pred_occ
    maxs = tf.reduce_max(a, axis=[1, 2, 3, 4], keepdims=True)
    return tf.reduce_max(tf.cast(maxs == a, tf.float32), axis=0, keepdims=True)


def are_chss_satisfied(pred_occ, chss):
    if chss is None:
        return True
    return tf.reduce_min(tf.reduce_max(chss * pred_occ, axis=(1, 2, 3, 4))) > 0.5


def get_most_wrong_free(pred_occ, known_free):
    """
    Returns a voxelgrid with only the most wrong known_free voxel
    Args:
        pred_occ:
        known_free:

    Returns:

    """
    return get_assumed_occ(pred_occ, known_free)


def denoise_pointcloud(pts, scale, origin, shape, threshold):
    vg = conversions.pointcloud_to_sparse_voxelgrid(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pts),
                                                    scale=scale, origin=origin, shape=shape)
    return conversions.sparse_voxelgrid_to_pointcloud(vg, scale=scale, origin=origin, threshold=threshold)


@dataclasses.dataclass
class BoxBounds:
    x_lower: float
    x_upper: float
    y_lower: float
    y_upper: float
    z_lower: float
    z_upper: float

    def is_in_box(self, pt):
        if not self.x_lower < pt[0] < self.x_upper:
            return False
        if not self.y_lower < pt[1] < self.y_upper:
            return False
        if not self.z_lower < pt[2] < self.z_upper:
            return False
        return True

def remove_points_outside_box(pts, box_bounds: BoxBounds):
    pts_filtered = np.array(list(pt for pt in pts if box_bounds.is_in_box(pt)))
    return pts_filtered



def satisfies_constraints(pred_occ, known_contact, known_free):
    return np.max(pred_occ * known_free) <= KNOWN_FREE_LIMIT and \
                tf.reduce_min(tf.boolean_mask(pred_occ, known_contact)) >= KNOWN_OCC_LIMIT


def enforce_contact(latent: tf.Variable, known_free, chss, pssnet: PSSNet, belief: ParticleBelief,
                    vg_pub: VoxelgridPublisher, verbose=True):
    success = True
    vg_pub.publish("aux", 0 * known_free)

    vg_pub.publish('predicted_occ', pssnet.decode(latent, apply_sigmoid=True))
    pred_occ = pssnet.decode(latent, apply_sigmoid=True)
    known_contact = get_assumed_occ(pred_occ, chss)
    # any_chs = tf.reduce_max(chss, axis=-1, keepdims=True)
    vg_pub.publish('known_free', known_free)
    if chss is None:
        vg_pub.publish('chs', known_free * 0)
    else:
        vg_pub.publish('chs', chss)

    if verbose:
        print()
    log_pdf = log_normal_pdf(latent, belief.latent_prior_mean, belief.latent_prior_logvar)
    quantile = belief.get_quantile(log_pdf)
    if verbose:
        print(f"Before optimization logprob: {log_pdf} ({quantile} quantile)")

    if satisfies_constraints(pred_occ, known_contact, known_free):
        if verbose:
            print(f"{Fore.GREEN}Particle satisfies constraints without any gradient needed{Fore.RESET}")
        return latent, True

    prev_loss = 0.0
    for i in range(GRADIENT_UPDATE_ITERATION_LIMIT):
        # single_free = get_most_wrong_free(pred_occ, known_free)

        loss = pssnet.grad_step_towards_output(latent, known_contact, known_free, belief)
        if verbose:
            print('\rloss: {}'.format(loss), end='')
        pred_occ = pssnet.decode(latent, apply_sigmoid=True)
        known_contact = get_assumed_occ(pred_occ, chss)
        vg_pub.publish('predicted_occ', pred_occ)
        vg_pub.publish('known_contact', known_contact)

        if satisfies_constraints(pred_occ, known_contact, known_free):
            if verbose:
                print(
                    f"\t{Fore.GREEN}All known free have less that {KNOWN_FREE_LIMIT} prob occupancy, "
                    f"and chss have value > {KNOWN_OCC_LIMIT}{Fore.RESET}")
            break

        if loss == prev_loss:
            if verbose:
                print("\tNo progress made. Accepting shape as is")
            success = False
            break
        prev_loss = loss

        if tf.math.is_nan(loss):
            if verbose:
                print("\tLoss is nan. There is a problem I am not addressing")
            success = False
            break

    else:
        success = False
        if verbose:
            print()
        if np.max(pred_occ * known_free) > KNOWN_FREE_LIMIT:
            if verbose:
                print("Optimization did not satisfy known freespace: ")
            vg_pub.publish("aux", pred_occ * known_free)
        if tf.reduce_min(tf.boolean_mask(pred_occ, known_contact)) < KNOWN_OCC_LIMIT:
            if verbose:
                print("Optimization did not satisfy assumed occ: ")
            vg_pub.publish("aux", (1 - pred_occ) * known_contact)
        if verbose:
            print('\tWarning, enforcing contact terminated due to max iterations, not actually satisfying contact')

    log_pdf = log_normal_pdf(latent, belief.latent_prior_mean, belief.latent_prior_logvar)
    quantile = belief.get_quantile(log_pdf)
    if verbose:
        print(f"Latent logprob {log_pdf} ({quantile} quantile)")
    return latent, success


def enforce_contact_ignore_latent_prior(latent: tf.Variable, known_free, chss, pssnet: PSSNet, belief: ParticleBelief,
                    vg_pub: VoxelgridPublisher):
    success = True
    vg_pub.publish("aux", 0 * known_free)

    vg_pub.publish('predicted_occ', pssnet.decode(latent, apply_sigmoid=True))
    pred_occ = pssnet.decode(latent, apply_sigmoid=True)
    known_contact = get_assumed_occ(pred_occ, chss)
    # any_chs = tf.reduce_max(chss, axis=-1, keepdims=True)
    vg_pub.publish('known_free', known_free)
    if chss is None:
        vg_pub.publish('chs', known_free * 0)
    else:
        vg_pub.publish('chs', chss)

    print()
    log_pdf = log_normal_pdf(latent, belief.latent_prior_mean, belief.latent_prior_logvar)
    quantile = belief.get_quantile(log_pdf)
    print(f"Before optimization logprob: {log_pdf} ({quantile} quantile)")

    if satisfies_constraints(pred_occ, known_contact, known_free):
        print(f"{Fore.GREEN}Particle satisfies constraints without any gradient needed{Fore.RESET}")
        return latent, True

    prev_loss = 0.0
    for i in range(GRADIENT_UPDATE_ITERATION_LIMIT):
        # single_free = get_most_wrong_free(pred_occ, known_free)

        loss = pssnet.grad_step_towards_output_ignore_latent_prior(latent, known_contact, known_free, belief)
        print('\rloss: {}'.format(loss), end='')
        pred_occ = pssnet.decode(latent, apply_sigmoid=True)
        known_contact = get_assumed_occ(pred_occ, chss)
        vg_pub.publish('predicted_occ', pred_occ)
        vg_pub.publish('known_contact', known_contact)

        if satisfies_constraints(pred_occ, known_contact, known_free):
            print(
                f"\t{Fore.GREEN}All known free have less that {KNOWN_FREE_LIMIT} prob occupancy, "
                f"and chss have value > {KNOWN_OCC_LIMIT}{Fore.RESET}")
            break

        if loss == prev_loss:
            print("\tNo progress made. Accepting shape as is")
            success = False
            break
        prev_loss = loss

        if tf.math.is_nan(loss):
            print("\tLoss is nan. There is a problem I am not addressing")
            success = False
            break

    else:
        success = False
        print()
        if np.max(pred_occ * known_free) > KNOWN_FREE_LIMIT:
            print("Optimization did not satisfy known freespace: ")
            vg_pub.publish("aux", pred_occ * known_free)
        if tf.reduce_min(tf.boolean_mask(pred_occ, known_contact)) < KNOWN_OCC_LIMIT:
            print("Optimization did not satisfy assumed occ: ")
            vg_pub.publish("aux", (1 - pred_occ) * known_contact)
        print('\tWarning, enforcing contact terminated due to max iterations, not actually satisfying contact')

    log_pdf = log_normal_pdf(latent, belief.latent_prior_mean, belief.latent_prior_logvar)
    quantile = belief.get_quantile(log_pdf)
    print(f"Latent logprob {log_pdf} ({quantile} quantile)")
    return latent, success