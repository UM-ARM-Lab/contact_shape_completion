from functools import lru_cache
from shape_completion_training.model import filepath_tools
import hjson


@lru_cache()
def get_config():
    fp = filepath_tools.get_shape_completion_package_path() / "config.hjson"
    with fp.open() as f:
        config = hjson.load(f)
    return config


def lookup_trial(trial_name):
    named_trials = {
        "YCB": "PSSNet_YCB/May_24_14-37-51_28829eda5b",
        # "AAB": "PSSNet_aab/May_27_11-10-21_226da3b642",
        "AAB": "PSSNet_aab/May_28_17-09-55_226da3b642",
        "shapenet_mugs": "PSSNet_shapenet_mugs/June_13_13-22-08_4ec9cf403f",
        "VAE_GAN_YCB": "VAE_GAN_YCB/June_15_16-32-49_31306d5b72",
        "VAE_GAN_mugs": "VAE_GAN_shapenet_mugs/June_14_14-22-56_31306d5b72",
        "VAE_GAN_aab": "VAE_GAN_aab/June_16_16-37-55_f17c7f7156"
    }
    if trial_name in named_trials:
        return named_trials[trial_name]
    return trial_name
