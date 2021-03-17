from functools import lru_cache
from shape_completion_training.model import filepath_tools
import hjson


@lru_cache()
def get_config():
    fp = filepath_tools.get_shape_completion_package_path() / "config.hjson"
    with fp.open() as f:
        config = hjson.load(f)
    return config
