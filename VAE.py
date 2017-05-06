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
            outputs = self.decoder(latent)
            self.tsample = tf.nn.sigmoid(outputs)

        with tf.variable_scope('model', reuse=True) as scope:
            gsample = tf.random_normal([self.conf.batch_size, self.conf.hidden_size])
            gsample = tf.reshape(gsample, [-1, 1, 1, self.conf.hidden_size])
            self.gsample = tf.nn.sigmoid(self.decoder(gsample))

        self.kl_loss = self.get_kl_loss(self.mean, self.stddev)
        self.ce_loss = self.get_CE_loss(outputs, self.X)

        self.loss = self.kl_loss + self.ce_loss

    def get_loss(self):
        return self.kl_loss, self.ce_loss, self.loss

    def get_tsample(self):
        return self.tsample

    def get_gsample(self):
        return self.gsample

    def get_CE_loss(self, logits, X):
        return tf.losses.sigmoid_cross_entropy(X, logits)

    def get_kl_loss(self, mean, stddev, offset=1e-8):
        '''
        compute KL divergence between N(mean, stddev^2) and N(0,I)
        '''
        return tf.reduce_mean(0.5*(tf.square(stddev) + tf.square(mean) - 1.0 - 2.0*tf.log(stddev+offset)))

    def log_marginal_likelihood_estimate(self): # binarized data
        '''
        compute log(p(x|z)) + log(z) - log(q(z|x)) once for the current batch
        '''
        x_mean = tf.reshape(self.X, [self.conf.batch_size, -1])
        x_sample = tf.reshape(self.tsample, [self.conf.batch_size, -1])
        return log_likelihood_bernoulli(x_mean, x_sample) +\
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
        # return tf.nn.sigmoid(outputs) # because input is scaled to [0,1]
        return outputs


