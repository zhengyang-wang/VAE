import numpy as np
import tensorflow as tf

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
