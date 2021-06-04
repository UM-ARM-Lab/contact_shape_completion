import tensorflow as tf


class ParticleBelief:
    def __init__(self):
        self.latent_prior_mean = None
        self.latent_prior_logvar = None
        self.particles = []
        self.quantiles_log_pdf = None

    def reset(self):
        self.latent_prior_mean = None
        self.latent_prior_logvar = None
        self.particles = []

    def get_quantile(self, log_pdf):
        try:
            gts = self.quantiles_log_pdf > log_pdf
            if tf.reduce_any(gts):
                return tf.where(gts)[0, 0].numpy()
            return len(self.quantiles_log_pdf)
        except Exception as e:
            print("I don't know what the error is")
            raise e


class Particle:
    def __init__(self):
        self.sampled_latent = None
        self.latent = None
        self.goal = None
        self.completion = None
        self.associated_chs_inds = []
        self.successful_projection = True
