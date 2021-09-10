import math

from colorama import Fore

from shape_completion_training.utils import tf_utils
from shape_completion_training.model.pssnet import PSSNet

tf_utils.set_gpu_with_lowest_memory()
import tensorflow as tf
from shape_completion_training.model import filepath_tools
from shape_completion_training.model.other_model_architectures.auto_encoder import AutoEncoder
from shape_completion_training.model.other_model_architectures.augmented_ae import Augmented_VAE
from shape_completion_training.model.other_model_architectures.voxelcnn import VoxelCNN
from shape_completion_training.model.other_model_architectures.conditional_vcnn import ConditionalVCNN
from shape_completion_training.model.other_model_architectures.ae_vcnn import AE_VCNN
from shape_completion_training.model.other_model_architectures.vae import VAE, VAE_GAN
from shape_completion_training.model.other_model_architectures.three_D_rec_gan import ThreeD_rec_gan
from shape_completion_training.model.flow import RealNVP
import progressbar
import datetime
import time


class ModelRunner:
    def __init__(self, training, group_name=None, trial_path=None, params=None, write_summary=True,
                 exists_required=False):
        """
        @type training: bool
        @param training: 
        @param group_name: 
        @param trial_path: 
        @param params: 
        @param write_summary:
        @param exists_required: If True, will fail if checkpoint does not already exist
        """
        self.side_length = 64
        self.num_voxels = self.side_length ** 3
        self.training = training

        self.trial_path, self.params = filepath_tools.create_or_load_trial(group_name=group_name,
                                                                           params=params,
                                                                           trial_path=trial_path,
                                                                           write_summary=write_summary)
        self.exists_required = exists_required
        self.group_name = self.trial_path.parts[-2]

        self.batch_size = 1 if not self.training else params['batch_size']

        self.train_summary_writer = tf.summary.create_file_writer((self.trial_path / "logs/train").as_posix())
        self.test_summary_writer = tf.summary.create_file_writer((self.trial_path / "logs/test").as_posix())

        if self.params['network'] == 'VoxelCNN':
            self.model = VoxelCNN(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'AutoEncoder':
            self.model = AutoEncoder(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'VAE':
            self.model = VAE(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'VAE_GAN':
            self.model = VAE_GAN(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'Augmented_VAE':
            self.model = Augmented_VAE(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'Conditional_VCNN':
            self.model = ConditionalVCNN(self.params, batch_size=self.batch_size)
        elif self.params['network'] == 'AE_VCNN':
            self.model = AE_VCNN(self.params, batch_size=self.batch_size)
        elif self.params['network'] == "RealNVP":
            self.model = RealNVP(hparams=self.params, batch_size=self.batch_size, training=training)
        elif self.params['network'] == "PSSNet" or \
                self.params['network'] == "NormalizingAE":  # NormalizingAE was legacy name
            self.model = PSSNet(self.params, batch_size=self.batch_size)
            self.model.flow = ModelRunner(training=False, trial_path=self.params['flow'],
                                          exists_required=True).model.flow
        elif self.params['network'] == "3D_rec_gan":
            self.model = ThreeD_rec_gan(self.params, batch_size=self.batch_size)
        else:
            raise Exception('Unknown Model Type')

        self.num_batches = None

        self.latest_ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                               epoch=tf.Variable(0),
                                               train_time=tf.Variable(0.0),
                                               net=self.model)
        self.latest_checkpoint_path = self.trial_path / "latest_checkpoints/"
        self.latest_checkpoint_manager = tf.train.CheckpointManager(self.latest_ckpt,
                                                                    self.latest_checkpoint_path.as_posix(),
                                                                    max_to_keep=1)
        self.num_batches = None

        self.best_ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                             epoch=tf.Variable(0),
                                             train_time=tf.Variable(0.0),
                                             best_key_metric_value=tf.Variable(10e10, dtype=tf.float32),
                                             net=self.model)
        self.best_checkpoint_path = self.trial_path / "best_checkpoint/"
        self.best_checkpoint_manager = tf.train.CheckpointManager(self.best_ckpt, self.best_checkpoint_path.as_posix(),
                                                                  max_to_keep=1)
        self.restore()

    def restore(self):
        status = self.best_ckpt.restore(self.best_checkpoint_manager.latest_checkpoint)

        # Suppress warning 
        if self.best_checkpoint_manager.latest_checkpoint:
            print(f"{Fore.CYAN}Restoring best checkpoint {self.best_checkpoint_manager.latest_checkpoint}{Fore.RESET}")
            status.assert_existing_objects_matched()
        elif self.exists_required:
            raise RuntimeError(f"Could not load required checkpoint for {self.trial_path}")

    def count_params(self):
        self.model.summary()

    def build_model(self, dataset):
        elem = next(dataset.get_training(self.params).batch(self.batch_size)).load()
        # elem = dataset.take(self.batch_size).batch(self.batch_size)
        tf.summary.trace_on(graph=True, profiler=False)
        self.model(elem)
        with self.train_summary_writer.as_default():
            tf.summary.trace_export(name='train_trace', step=self.latest_ckpt.step.numpy())

        # tf.keras.utils.plot_model(self.model, (self.trial_path / 'network.png').as_posix(),
        #                           show_shapes=True)

    def write_summary(self, summary_dict):
        with self.train_summary_writer.as_default():
            for k in summary_dict:
                tf.summary.scalar(k, summary_dict[k].numpy(), step=self.latest_ckpt.step.numpy())

    def train_batch(self, dataset, summary_period=10):
        if self.num_batches is not None:
            max_size = str(self.num_batches)
        else:
            max_size = '???'

        widgets = [
            '  ', progressbar.Counter(), '/', max_size,
            ' ', progressbar.Variable("Loss"), ' ',
            progressbar.Bar(),
            ' [', progressbar.Variable("TrainTime"), '] ',
            ' (', progressbar.ETA(), ') ',
        ]

        with progressbar.ProgressBar(widgets=widgets, max_value=self.num_batches) as bar:
            self.num_batches = 0
            t0 = time.time()
            total_loss = 0
            for batch in dataset:
                self.num_batches += 1
                self.latest_ckpt.step.assign_add(1)
                data = batch.load()

                _, ret = self.model.train_step(data)
                time_str = str(datetime.timedelta(seconds=int(self.latest_ckpt.train_time.numpy())))
                total_loss += ret['loss'].numpy()
                bar.update(self.num_batches, Loss=ret['loss'].numpy(),
                           TrainTime=time_str)
                if self.num_batches % summary_period == 0:
                    self.write_summary(ret)
                self.latest_ckpt.train_time.assign_add(time.time() - t0)
                t0 = time.time()
        avg_loss = total_loss / self.num_batches

        if math.isnan(avg_loss):
            raise RuntimeError("Loss is NaN. Probably an unrecoverable situation")

        if avg_loss < self.best_ckpt.best_key_metric_value:
            self.best_ckpt.best_key_metric_value.assign(avg_loss)
            save_path = self.best_checkpoint_manager.save()
            print(f"{Fore.GREEN}Saved new best checkpoint {save_path}{Fore.RESET}")

        save_path = self.latest_checkpoint_manager.save()
        print(f"Saved checkpoint for step {int(self.latest_ckpt.step)}: {save_path}")
        print("Avg loss {:1.3f}".format(avg_loss))

    def train(self, dataset):
        self.build_model(dataset)
        self.count_params()
        # dataset = dataset.shuffle(10000)
        # batched_ds = dataset.batch(self.batch_size, drop_remainder=True).prefetch(64)
        # batched_ds = dataset.batch(self.batch_size).prefetch(64)

        num_epochs = 1000
        while self.latest_ckpt.epoch < num_epochs:
            self.latest_ckpt.epoch.assign_add(1)
            print('')
            print('==  Epoch {}/{}  '.format(self.latest_ckpt.epoch.numpy(), num_epochs) + '=' * 25
                  + ' ' + self.group_name + ' ' + '=' * 20)
            self.train_batch(dataset.get_training(self.params).batch(self.batch_size))
            print('=' * 48)

    def train_and_test(self, dataset):
        train_ds = dataset
        self.train(train_ds)
        self.count_params()

    def evaluate(self, dataset):
        self.model.evaluate(dataset.batch(self.batch_size))
