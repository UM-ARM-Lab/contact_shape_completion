from unittest import TestCase
import numpy as np
from shape_completion_training.utils.tf_utils import log_normal_pdf, sample_gaussian


class Test(TestCase):
    def test_log_normal_pdf_in_one_D(self):
        # mean = np.array([[0.0]], dtype=np.float32)
        # var = np.array([[1.0]], dtype=np.float32)

        def p(sample, mean, var):
            u = np.array([[mean]], dtype=np.float32)
            sigma = np.array([[var]], dtype=np.float32)
            return np.exp(log_normal_pdf(sample, u, np.log(sigma)).numpy()[0])

        self.assertAlmostEqual(p(0, 0, 1), .3989, delta=.001)
        self.assertAlmostEqual(p(0, 0, 2), .28209, delta=.001)
        self.assertAlmostEqual(p(1, 0, 1), .24197, delta=.001)
        self.assertAlmostEqual(p(1, 0, 2), .21970, delta=.001)
        self.assertAlmostEqual(p(0, 1, 2), .21970, delta=.001)

    def test_relative_probability_in_one_D(self):
        def p(dist_from_mean, var):
            u = np.array([[0]], dtype=np.float32)
            root_sigma = np.array([[var]], dtype=np.float32)
            return np.exp(log_normal_pdf(dist_from_mean, u, np.log(root_sigma)).numpy()[0])

        rel_one_at_one = p(1, 1) / p(0, 1)
        rel_two_at_two = p(2, 4) / p(0, 4)

        self.assertAlmostEqual(rel_two_at_two, rel_one_at_one, delta=.001)

    def test_log_normal_pdf_in_two_D(self):
        # mean = np.array([[0.0]], dtype=np.float32)
        # var = np.array([[1.0]], dtype=np.float32)

        def p(sample, mean, var):
            u = np.array([mean], dtype=np.float32)
            sigma = np.array([var], dtype=np.float32)
            return np.exp(log_normal_pdf(sample, u, np.log(sigma)).numpy()[0])

        # Naming schema is {num_variances_from_mean}_for_{var}
        # These are the (1-d) probability density for a gaussian as a function of variance and num_variances from mean
        zero_for_one = .3989
        one_for_one = .24197
        zero_for_two = .28209
        one_for_two = .21970

        self.assertAlmostEqual(p([0, 0], [0, 0], [1, 1]), zero_for_one ** 2, delta=.001)
        self.assertAlmostEqual(p([0, 1], [0, 1], [1, 1]), zero_for_one ** 2, delta=.001)
        self.assertAlmostEqual(p([0, 0], [0, 1], [1, 1]), zero_for_one * one_for_one, delta=.001)
        self.assertAlmostEqual(p([0, 0], [1, 0], 2), one_for_two * zero_for_two, delta=.001)
        self.assertAlmostEqual(p([0, 1], [1, 0], 2), one_for_two * one_for_two, delta=.001)

    def test_two_hundred_D(self):
        def logp(sample, mean, var):
            u = np.array([mean], dtype=np.float32)
            sigma = np.array([var], dtype=np.float32)
            return log_normal_pdf(sample, u, np.log(sigma)).numpy()[0]

        # Naming schema is {num_variances_from_mean}_for_{var}
        # These are the (1-d) probability density for a gaussian as a function of variance and num_variances from mean
        zero_for_one = .3989
        one_for_one = .24197
        zero_for_two = .28209
        one_for_two = .21970

        mean = np.zeros(200)
        samples = np.zeros(200)
        var = np.ones(200)
        self.assertAlmostEqual(logp(samples, mean, var), 200 * np.log(zero_for_one), delta=0.1)
        print(200 * np.log(zero_for_one))

    def test_sample_gaussian(self):
        def sample(mean, var):
            u = np.array([[mean]], dtype=np.float32)
            sigma = np.array([[var]], dtype=np.float32)
            return sample_gaussian(u, np.log(sigma)).numpy()[0, 0]

        def repeated_gaussian_sampling_test(mean: float, var: float, allowed_delta: float, allowed_delta_all: float):
            samples = []
            for i in range(100):
                samples.append(sample(mean, var))
                self.assertAlmostEqual(samples[-1], mean, delta=allowed_delta)
            self.assertAlmostEqual(float(np.mean(samples)), mean, delta=allowed_delta_all)
            # print(np.mean(samples))

        repeated_gaussian_sampling_test(0, 1, 4, .5)
        repeated_gaussian_sampling_test(5, 1, 4, .5)
        repeated_gaussian_sampling_test(0, .1 ** 2, .4, .05)
        repeated_gaussian_sampling_test(5, .1 ** 2, .4, .05)
        # for i in range(100):
        #     self.assertAlmostEqual(sample(0, 1), 0, delta=3.5)
        #     self.assertAlmostEqual(sample(5, 1), 5, delta=3.5)
        #     self.assertAlmostEqual(sample(0, .1), 0, delta=.35)
        #     self.assertAlmostEqual(sample(5, .1), 5, delta=.35)
