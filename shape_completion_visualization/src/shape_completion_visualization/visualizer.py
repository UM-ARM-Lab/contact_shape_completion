import argparse

import rospy
from rviz_text_selection_panel_msgs.msg import TextSelectionOptions
from std_msgs.msg import String

from shape_completion_training.model.model_runner import ModelRunner
# from shape_completion_training.utils.shapenet_storage import ShapenetDatasetSupervisor
from shape_completion_training.utils.dataset_supervisor import get_dataset_supervisor
# from shape_completion_visualization.shape_selection import send_display_names
from shape_completion_visualization.voxelgrid_publisher import VoxelgridPublisher


def parse_visualizer_command_line_args(**args):
    parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
    parser.add_argument('--sample', help='foo help', action='store_true')
    parser.add_argument('--use_best_iou', help='foo help', action='store_true')
    parser.add_argument('--publish_each_sample', help='foo help', action='store_true')
    parser.add_argument('--fit_to_particles', help='foo help', action='store_true')
    parser.add_argument('--publish_nearest_plausible', help='foo help', action='store_true')
    parser.add_argument('--publish_nearest_sample', help='foo help', action='store_true')
    parser.add_argument('--multistep', action='store_true')
    parser.add_argument('--trial')
    parser.add_argument('--dataset', default='shapenet_mugs')
    parser.add_argument('--publish_closest_train', action='store_true')
    parser.add_argument('--enforce_contact', action='store_true')

    return parser.parse_args()


def send_display_names(names_list, callback):
    """
    Sends a list of names to interact with rviz_text_selection_panel
    @param names_list: list of names
    @param callback: python callback function of form `def fn(ind, name)`
    @return:
    """
    options_pub = rospy.Publisher('shapenet_options', TextSelectionOptions, queue_size=1)
    selection_map = {}
    tso = TextSelectionOptions()
    for i, name in enumerate(names_list):
        selection_map[name] = i
        tso.options.append(name)
    i = 1
    while options_pub.get_num_connections() == 0:
        i += 1
        if i % 10 == 0:
            rospy.loginfo("Waiting for options publisher to connect to topic: {}".format('shapenet_options'))
        rospy.sleep(0.1)
    options_pub.publish(tso)

    def callback_fn(str_msg):
        return callback(selection_map[str_msg.data], str_msg)

    sub = rospy.Subscriber('/shapenet_selection', String, callback_fn)
    rospy.sleep(0.1)

    return sub


class Visualizer:
    def __init__(self, params, **args):
        self.vg_pub = VoxelgridPublisher()
        self.selection_sub = None
        self.model_runner = None
        self.dataset_supervisor = None
        self.params = params

        self.train_or_test = args['train_or_test'] if 'train_or_test' in args else 'train'

        if args['dataset']:
            self.load_dataset(args['dataset'])

        if args['trial']:
            self.load_model(args['trial'])

    def load_dataset(self, name):
        self.dataset_supervisor = get_dataset_supervisor(name)

        md = self.dataset_supervisor.train_md if self.train_or_test == "train" else self.dataset_supervisor.test_md
        names = self.dataset_supervisor.ind_for_train_id if self.train_or_test == "train" \
            else self.dataset_supervisor.ind_for_test_id
        self.selection_sub = send_display_names(names.keys(), self.publish_selection)

    def publish_selection(self, ind, unique_id):
        # print(f"Publishing {ind}, {unique_id}")
        elem = self.dataset_supervisor.get_element(unique_id.data, self.params).load()
        self.vg_pub.publish_elem(elem)

        if self.model_runner is None:
            return

        inference = self.model_runner.model(elem)
        self.vg_pub.publish_inference(inference)

    def load_model(self, trial):
        self.model_runner = ModelRunner(training=False, trial_path=trial)
