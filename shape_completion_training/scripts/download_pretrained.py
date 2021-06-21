#! /usr/bin/env python
"""
Downloads pretrained models, preprocess datasets, and precomputed plausible sets
"""
from pathlib import Path

import rospy

from shape_completion_training.model import filepath_tools

try:
    from google_drive_downloader import GoogleDriveDownloader as gdd
except Exception as e:
    print("Failed to import gdd.")
    print("Either `pip install googledrivedownloader`")
    print("or manually download checkpoint from google drive")
    raise e

data_id = '1iqpDvhAGNA6SahrEsHtQYQw11MYSaJnx'
trials_id = '15AZgbTf-rN4b47l0fGGqpsDHnJx7sqZz'

if __name__ == "__main__":
    rospy.init_node("pretrained_downloader")
    pkg_path = Path(filepath_tools.get_shape_completion_package_path())
    print("Downloading preprocessed datasets")
    gdd.download_file_from_google_drive(file_id=data_id,
                                        dest_path=(pkg_path / 'data.zip').as_posix(),
                                        unzip=True)
    print("Downloading pretrained models")
    gdd.download_file_from_google_drive(file_id=trials_id,
                                        dest_path=(pkg_path / 'trials.zip').as_posix(),
                                        unzip=True)
