import argparse

from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.utils.shapenet_storage import ShapenetDatasetSupervisor, get_unique_name
from shape_completion_visualization.shape_selection import send_display_names
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


class Visualizer:
    def __init__(self, **args):
        self.vg_pub = VoxelgridPublisher()
        self.selection_sub = None
        self.model_runner = None
        self.dataset_supervisor = None

        self.train_or_test = args['train_or_test'] if 'train_or_test' in args else 'train'

        if args['dataset']:
            self.load_dataset(args['dataset'])

        if args['trial']:
            self.load_model(args['trial'])

    def load_dataset(self, name):
        self.dataset_supervisor = ShapenetDatasetSupervisor(name)

        md = self.dataset_supervisor.train_md if self.train_or_test == "train" else self.dataset_supervisor.test_md
        names = self.dataset_supervisor.ind_for_train_id if self.train_or_test == "train" \
            else self.dataset_supervisor.ind_for_test_id
        self.selection_sub = send_display_names(names.keys(), self.publish_selection)

    def publish_selection(self, ind, unique_id):
        # print(f"Publishing {ind}, {unique_id}")
        elem = self.dataset_supervisor.get_element(unique_id.data).load()
        self.vg_pub.publish_elem(elem)

        if self.model_runner is None:
            return

        inference = self.model_runner.model(elem)
        self.vg_pub.publish_inference(inference)


    def load_model(self, trial):
        self.model_runner = ModelRunner(training=False, trial_path=trial)
