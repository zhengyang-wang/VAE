import numpy as np
import tensorflow as tf
from ops import *

class VAE(object):

    def __init__(self, X, config):
        self.X = X
        self.conf = config
        self.build_VAE()

    def build_VAE(self):
        with tf.variable_scope('model') as scope:
            encoding = self.encoder(self.X, self.conf.hidden_size*2)

            self.mean = encoding[:, :self.conf.hidden_size]
            self.stddev = tf.sqrt(tf.exp(encoding[:, self.conf.hidden_size:self.conf.hidden_size*2]))

            epsilon = tf.random_normal([self.conf.batch_size, self.conf.hidden_size])
            self.latent_sample = self.mean + tf.multiply(self.stddev, epsilon)

            latent = tf.reshape(self.latent_sample, [-1, 1, 1, self.conf.hidden_size])
            self.tsample = self.decoder(latent)

        with tf.variable_scope('model', reuse=True) as scope:
            gsample = tf.random_normal([self.conf.batch_size, self.conf.hidden_size])
            gsample = tf.reshape(gsample, [-1, 1, 1, self.conf.hidden_size])
            self.gsample = self.decoder(gsample)

        self.kl_loss = self.get_kl_loss(self.mean, self.stddev)
        self.l2_loss = self.get_l2_loss(self.tsample, self.X, self.conf.sigma)

        self.loss = self.kl_loss + self.l2_loss

    def get_loss(self):
        return self.kl_loss, self.l2_loss, self.loss

    def get_tsample(self):
        return self.tsample

    def get_gsample(self):
        return self.gsample

    def get_l2_loss(self, tsample, X, sigma):
        '''
        compute ||tsample-X||^2 * (1/sigma^2)
        '''
        l2 = tf.losses.mean_squared_error(tsample, X)
        one_over_sigma_square = tf.div(1.0, tf.square(sigma))
        return tf.scalar_mul(one_over_sigma_square, l2)

    def get_kl_loss(self, mean, stddev, offset=1e-8):
        '''
        compute KL divergence between N(mean, stddev^2) and N(0,I)
        '''
        return tf.reduce_mean(0.5*(tf.square(stddev) + tf.square(mean) - 1.0 - 2.0*tf.log(stddev+offset)))

    def log_marginal_likelihood_estimate(self):
        '''
        compute log(p(x|z)) + log(z) - log(q(z|x)) once for the current batch
        '''
        x_mean = tf.reshape(self.X, [self.conf.batch_size, -1])
        x_sample = tf.reshape(self.tsample, [self.conf.batch_size, -1])
        x_sigma = tf.multiply(self.conf.sigma, tf.ones(x_mean.shape))
        return log_likelihood_gaussian(x_sample, x_mean, x_sigma) +\
                log_likelihood_prior(self.latent_sample) -\
                log_likelihood_gaussian(self.latent_sample, self.mean, self.stddev)

    def encoder(self, inputs, num_outputs):
        '''
        inputs: N*H*W*C
        num_outputs: integer number n

        return: N*n
        '''
        outputs = conv2d_bn_activated(inputs, 32, 5, 2, 'SAME')
        outputs = conv2d_bn_activated(outputs, 64, 5, 2, 'SAME')
        outputs = conv2d_bn_activated(outputs, 128, 5, 2, 'VALID')
        # outputs = tf.contrib.layers.dropout(outputs, keep_prob=0.9)
        outputs = tf.contrib.layers.flatten(outputs)
        return tf.contrib.layers.fully_connected(outputs, num_outputs, activation_fn=tf.nn.elu)

    def decoder(self, inputs):
        '''
        inputs: N*d*d*c

        return: N*H*W*C
        '''
        outputs = conv2d_t_bn_activated(inputs, 128, 3, 1, 'VALID')
        outputs = conv2d_t_bn_activated(outputs, 64, 5, 1, 'VALID')
        outputs = conv2d_t_bn_activated(outputs, 32, 5, 2, 'SAME')
        outputs = conv2d_t_bn(outputs, self.conf.channel, 5, 2, 'SAME')
        return tf.nn.sigmoid(outputs) # because input is scaled to [0,1]


