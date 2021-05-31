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
    }
    if trial_name in named_trials:
        return named_trials[trial_name]
    return trial_name
