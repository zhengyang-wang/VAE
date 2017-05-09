import numpy as np
import tensorflow as tf

def parzen_cpu_batch(x_batch, samples, sigma, batch_size, num_of_samples, data_size):
    '''
    x_batch:    a data batch (batch_size, data_size), data_size = h*w*c for images
    samples:    generated data (num_of_samples, data_size)
    sigma:      standard deviation (float32)
    '''

    # a=(x-x_i)/sigma, use broadcast
    x = x_batch.reshape((batch_size, 1, data_size))
    mu = samples.reshape((1, num_of_samples, data_size))
    a = (x - mu)/sigma # (batch_size, num_of_samples, data_size)

    # sum -0.5*a^2
    tmp = -0.5*(a**2).sum(2) # (batch_size, num_of_samples)

    # log_mean_exp trick
    max_ = np.amax(tmp, axis=1, keepdims=True) # (batch_size, 1)
    E = max_ + np.log(np.mean(np.exp(tmp - max_), axis=1, keepdims=True)) # (batch_size, 1)

    # Z = dim * log(sigma * sqrt(2*pi)), dim = data_size
    Z = data_size * np.log(sigma * np.sqrt(np.pi * 2))

    return E-Z

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
