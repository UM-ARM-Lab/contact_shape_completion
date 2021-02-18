from unittest import TestCase
import numpy as np
from shape_completion_training.utils.tf_utils import log_normal_pdf, sample_gaussian


class Test(TestCase):
    def test_log_normal_pdf(self):
        # mean = np.array([[0.0]], dtype=np.float32)
        # var = np.array([[1.0]], dtype=np.float32)

        def p(sample, mean, var):
            u = np.array([[mean]], dtype=np.float32)
            sigma = np.array([[var]], dtype=np.float32)
            return np.exp(log_normal_pdf(sample, u, np.log(sigma)).numpy()[0])

        self.assertAlmostEqual(p(0, 0, 1), .3989, delta=.001)
        self.assertAlmostEqual(p(0, 0, 2), .28209, delta=.001)
        self.assertAlmostEqual(p(1, 0, 2), .21970, delta=.001)
        self.assertAlmostEqual(p(0, 1, 2), .21970, delta=.001)
        # self.fail()

    def test_sample_gaussian(self):
        def sample(mean, var):
            u = np.array([[mean]], dtype=np.float32)
            sigma = np.array([[var]], dtype=np.float32)
            return sample_gaussian(u, np.log(sigma)).numpy()[0,0]

        def repeated_gaussian_sampling_test(mean: float, var: float, allowed_delta: float, allowed_delta_all: float):
            samples = []
            for i in range(100):
                samples.append(sample(mean, var))
                self.assertAlmostEqual(samples[-1], mean, delta=allowed_delta)
            self.assertAlmostEqual(float(np.mean(samples)), mean, delta=allowed_delta_all)
            print(np.mean(samples))

        repeated_gaussian_sampling_test(0, 1, 4, .5)
        repeated_gaussian_sampling_test(5, 1, 4, .5)
        repeated_gaussian_sampling_test(0, .1**2, .4, .05)
        repeated_gaussian_sampling_test(5, .1**2, .4, .05)
        # for i in range(100):
        #     self.assertAlmostEqual(sample(0, 1), 0, delta=3.5)
        #     self.assertAlmostEqual(sample(5, 1), 5, delta=3.5)
        #     self.assertAlmostEqual(sample(0, .1), 0, delta=.35)
        #     self.assertAlmostEqual(sample(5, .1), 5, delta=.35)


