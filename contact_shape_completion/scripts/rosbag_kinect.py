import rospy
from shape_completion_training.model.filepath_tools import get_shape_completion_package_path
import argparse
import rosbag
import importlib


def _get_rosbag_path():
    return (get_shape_completion_package_path() / "../rosbag/").resolve()


def record_files():
    rosbag_topic_list_fp = _get_rosbag_path() / "camera_topics.txt"

    topics = []
    with rosbag_topic_list_fp.open() as f:
        for line in f.read().split("\n"):
            topics.append(line)

    print("Copy and run this in a sourced command line after roscd contact_shape_completion & cd rosbag\n\n")
    print(f"rosbag record -O camera {' '.join(topics)}\n\n")


def play_rosbag(filename):
    pass


class RosbagPlayer:
    def __init__(self, bagfile):
        fp = (_get_rosbag_path() / bagfile).with_suffix('.bag')
        if not fp.exists():
            raise FileNotFoundError(f"Rosbag file {fp} does not exist")

        self.bagfile = fp
        self.publishers = dict()

        self.start_publishers()

    def start_publishers(self):
        for topic, msg, t in rosbag.Bag(self.bagfile.as_posix()).read_messages():
            # print(f'{topic=}, {msg=}, {t=}')
            if topic in self.publishers:
                continue
            print(f"Starting publisher on topic {topic}")
            type = msg._type.split('/')
            # eval(f"import {type}")
            pkg = '.'.join(type[:-1]) + ".msg"
            i = importlib.import_module(pkg)
            self.publishers[topic] = rospy.Publisher(topic, eval(f"i.{type[-1]}"), queue_size=10)
        rospy.sleep(0.5)  # Wait for publishers to connect

    def play(self):
        print("Playing...")
        t0 = None
        t_play_start = rospy.Time().now()
        for topic, msg, t in rosbag.Bag(self.bagfile.as_posix()).read_messages():
            if t0 is None:
                t0 = t

            while rospy.Time().now() - t_play_start < t - t0:
                rospy.sleep(0.01)

            pub = self.publishers[topic]

            # now = rospy.Time.now()
            # now = t_play_start - t0
            # now.secs += t.secs
            offset = t_play_start - t0
            if topic == "/tf":
                for m in msg.transforms:
                    m.header.stamp.secs += offset.secs
                    m.header.stamp.nsecs += offset.nsecs
            elif msg._has_header:
                # print(topic)
                msg.header.stamp.secs += offset.secs
                msg.header.stamp.nsecs += offset.nsecs

            pub.publish(msg)

        # rospy.spin()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="Prints a terminal 'rosbag' command for recording")
    parser.add_argument("--play", help="Plays the relevant rosbag")

    args = parser.parse_args()

    if args.record:
        print("")
        record_files()
    if args.play:
        rospy.init_node("rosbag_player")
        print(f"playing rosbag")
        # play_rosbag(args.play)
        rb_player = RosbagPlayer(args.play)
        rb_player.play()
