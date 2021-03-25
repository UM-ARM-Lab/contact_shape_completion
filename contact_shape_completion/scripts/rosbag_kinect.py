#!/usr/bin/env python
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
        self.static_tf_msg = None
        self.extract_static_tfs()

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
            if topic == "/tf_static":
                self.publishers[topic] = rospy.Publisher(topic, eval(f"i.{type[-1]}"), queue_size=10, latch=True)
                continue

            self.publishers[topic] = rospy.Publisher(topic, eval(f"i.{type[-1]}"), queue_size=10)
        rospy.sleep(0.5)  # Wait for publishers to connect

    def extract_static_tfs(self):
        for topic, msg, t in rosbag.Bag(self.bagfile.as_posix()).read_messages():
            if topic != "/tf_static":
                continue
            if self.static_tf_msg is None:
                self.static_tf_msg = msg
            else:
                self.static_tf_msg.transforms += msg.transforms

    def play(self, repeat=False):
        self._play_helper(is_repeat=False)

        if repeat:
            while not rospy.is_shutdown():
                self._play_helper(is_repeat=True)

    def _play_helper(self, is_repeat):
        print("Playing...")
        t0 = None
        t_play_start = rospy.Time().now()

        if self.static_tf_msg is not None and not is_repeat:
            self.publishers['/tf_static'].publish(self.static_tf_msg)

        for topic, msg, t in rosbag.Bag(self.bagfile.as_posix()).read_messages():
            if rospy.is_shutdown():
                return

            if t0 is None:
                t0 = t

            while rospy.Time().now() - t_play_start < t - t0:
                rospy.sleep(0.001)

            pub = self.publishers[topic]
            offset = t_play_start - t0

            if topic == "/tf_static":
                continue

            if topic == "/tf":
                for m in msg.transforms:
                    m.header.stamp.secs += offset.secs
                    m.header.stamp.nsecs += offset.nsecs
                    # m.header.stamp = rospy.Time().now()
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
    parser.add_argument("--repeat", action="store_true", help="Repeats the bag indefinitely")

    args = parser.parse_args()

    if args.record:
        print("")
        record_files()
    if args.play:
        rospy.init_node("rosbag_player")
        print(f"playing rosbag")
        # play_rosbag(args.play)
        rb_player = RosbagPlayer(args.play)
        rb_player.play(repeat=args.repeat)
        # if args.repeat:
        #     while not rospy.is_shutdown():
        #         rb_player.play()
