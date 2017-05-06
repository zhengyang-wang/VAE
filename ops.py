import numpy as np
import tensorflow as tf

log2pi = tf.constant(np.log(2*np.pi), tf.float32)

def log_likelihood_bernoulli(sample, mean):
    return tf.reduce_sum(sample * tf.log(mean) + (1 - sample) * tf.log(1 - mean), axis=1)

def log_likelihood_gaussian(sample, mean, sigma):
    '''
    compute log(sample~Gaussian(mean, sigma^2))
    '''
    return -log2pi*tf.cast(sample.shape[1].value, tf.float32)/2 -\
            tf.reduce_sum(tf.square((sample-mean)/sigma) + 2*tf.log(sigma), 1)/2

def log_likelihood_prior(sample):
    '''
    compute log(sample~Gaussian(0, I))
    '''
    return -log2pi*tf.cast(sample.shape[1].value, tf.float32)/2 -\
            tf.reduce_sum(tf.square(sample), 1)/2

# def lrelu(x, alpha=0.1):
#     return tf.maximum(alpha * x, x)

# def conv2d_bn_activated(inputs, num_outputs, kernel_size, stride, padding):
#     conv = tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride, padding,
#                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
#                                     activation_fn=None)
#     conv = tf.contrib.layers.batch_norm(conv)
#     conv = lrelu(conv)
#     return conv

# def conv2d_t_bn_activated(inputs, num_outputs, kernel_size, stride, padding):
#     conv = tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size, stride, padding,
#                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
#                                     activation_fn=None)
#     conv = tf.contrib.layers.batch_norm(conv)
#     conv = lrelu(conv)
#     return conv

# def conv2d_bn(inputs, num_outputs, kernel_size, stride, padding):
#     conv = tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride, padding,
#                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
#                                     activation_fn=None)
#     conv = tf.contrib.layers.batch_norm(conv)
#     return conv

# def conv2d_t_bn(inputs, num_outputs, kernel_size, stride, padding):
#     conv = tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size, stride, padding,
#                                     weights_initializer=tf.random_normal_initializer(stddev=0.02),
#                                     weights_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5),
#                                     activation_fn=None)
#     conv = tf.contrib.layers.batch_norm(conv)
#     return conv

def conv2d_bn(inputs, num_outputs, kernel_size, stride, padding):
    return tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride, padding,
                                    activation_fn=None,
                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                    normalizer_params={'scale': True})

def conv2d_t_bn(inputs, num_outputs, kernel_size, stride, padding):
    return tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size, stride, padding,
                                    activation_fn=None,
                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                    normalizer_params={'scale': True})

def conv2d_bn_activated(inputs, num_outputs, kernel_size, stride, padding):
    return tf.contrib.layers.conv2d(inputs, num_outputs, kernel_size, stride, padding,
                                    activation_fn=tf.nn.elu,
                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                    normalizer_params={'scale': True})

def conv2d_t_bn_activated(inputs, num_outputs, kernel_size, stride, padding):
    return tf.contrib.layers.conv2d_transpose(inputs, num_outputs, kernel_size, stride, padding,
                                    activation_fn=tf.nn.elu,
                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                    normalizer_params={'scale': True})
