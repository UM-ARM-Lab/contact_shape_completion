import rospkg
import pickle

from shape_completion_training.model.observation_model import observation_likelihood_geometric_mean, out_of_range_count
from shape_completion_training.utils import data_tools
from shape_completion_training.voxelgrid import fit, conversions
from shape_completion_training.model.utils import memoize
import progressbar


def _get_path(dataset_name, identifier=""):
    r = rospkg.RosPack()
    trial_path = data_tools.get_dataset_path(dataset_name) / ("plausibility" + identifier + ".pkl")
    return trial_path.as_posix()


@memoize
def load_plausibilities(dataset_name):
    with open(_get_path(dataset_name), "rb") as f:
        return pickle.load(f)


def save_plausibilities(plausibilities_dict, dataset_name, identifier=""):
    with open(_get_path(dataset_name, identifier), "wb") as f:
        pickle.dump(plausibilities_dict, f)


def get_plausibilities_for(shape_name, dataset_name):
    fits = load_plausibilities(dataset_name)[shape_name]
    valid_fits = [(k, v["T"], v["observation_probability"], v["out_of_range_count"])
                  for k, v in fits.items()
                  if v["out_of_range_count"] == 0]
    return valid_fits


def compute_icp_fit_dict(metadata, params):
    return compute_partial_icp_fit_dict(metadata, metadata, params)


def compute_partial_icp_fit_dict(reference_metadata, other_metadata, params):
    best_fits = {}
    num_shapes = 0
    for i in reference_metadata:
        num_shapes += 1

    reference_ds = data_tools.load_voxelgrids(reference_metadata)
    reference_ds = data_tools.preprocess_test_dataset(reference_ds, params)

    other_ds = data_tools.load_voxelgrids(other_metadata)
    other_ds = data_tools.preprocess_test_dataset(reference_ds, params)

    widgets = [
        '  ', progressbar.Counter(), '/', str(num_shapes),
        ' ', progressbar.Variable("CurrentShape"), ' ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]

    with progressbar.ProgressBar(widgets=widgets, max_value=num_shapes) as bar:
        for i, reference in reference_ds.enumerate():
            # bar.update(i.numpy())
            bar.update(i.numpy(), CurrentShape=data_tools.get_unique_name(reference))

            best_fit_for_reference = {}

            for other in other_ds:
                other_name = data_tools.get_unique_name(other)
                # print("    Fitting: {}".format(other_name))
                T = fit.icp_transform(other['known_occ'], reference['known_occ'], scale=0.01)
                fitted = conversions.transform_voxelgrid(other['gt_occ'], T, scale=0.01)
                p = observation_likelihood_geometric_mean(reference['gt_occ'], fitted,
                                                          std_dev_in_voxels=2)
                oob = out_of_range_count(reference['gt_occ'], fitted, width=4)

                # To make storage feasible, do not store fits that will not be used later
                if oob > 10:
                    continue
                best_fit_for_reference[other_name] = {"T": T, "observation_probability": p,
                                                      "out_of_range_count": oob}

            best_fits[data_tools.get_unique_name(reference)] = best_fit_for_reference
    return best_fits
