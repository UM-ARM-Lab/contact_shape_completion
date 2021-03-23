from shape_completion_training.model.filepath_tools import get_shape_completion_package_path


def record_files():
    rosbag_topic_list_fp = get_shape_completion_package_path() / "../rosbag/camera_topics.txt"
    topics = []
    with rosbag_topic_list_fp.open() as f:
        for line in f.read().split("\n"):
            topics.append(line)

    print("Copy and run this in a sourced command line after roscd contact_shape_completion & cd rosbag\n\n")
    print(f"rosbag record -O camera {' '.join(topics)}\n\n")


if __name__ == "__main__":
    record_files()
