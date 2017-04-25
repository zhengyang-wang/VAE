import os
import tensorflow as tf

from model import Model

flags = tf.flags
# training
flags.DEFINE_integer('max_epochs', 10, 'maximum number of epochs')
flags.DEFINE_integer('validation_interval', 550, 'number of iterations for validation')
flags.DEFINE_integer('save_interval', 10, 'number of epochs for saving')
flags.DEFINE_integer("updates_per_epoch", 550, "number of updates per epoch")
flags.DEFINE_integer('summary_step', 50, 'number of steps to save the summary')
flags.DEFINE_float('learning_rate', 1e-2, 'learning rate')
# data
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('channel', 1, 'number of channels')
flags.DEFINE_integer('height', 28, 'image height')
flags.DEFINE_integer('width', 28, 'image width')
flags.DEFINE_integer('hidden_size', 128, 'dimension of latent variable')
flags.DEFINE_float('sigma', 1.0, 'hyperparameter sigma')
flags.DEFINE_string('data_dir', '/tempspace/zwang6/mnist', 'data directory')
# debug
flags.DEFINE_string('model_name', 'model', 'model filename')
flags.DEFINE_string('logdir', 'log', 'log directory')
flags.DEFINE_string('modeldir', 'model', 'model directory')
# flags.DEFINE_string('logfile', 'log.txt', 'log filename')
# flags.DEFINE_boolean('is_train', True, 'Training or valid/test')
# flags.DEFINE_boolean('is_valid', True, 'Validation or testing')
flags.DEFINE_integer('random_seed', 1, 'random seed') # int(time.time())
flags.DEFINE_integer('reload_epoch', 0, 'Reload epoch')
# flags.DEFINE_integer('model_num', 50000, 'which model for valid/test')

# encoder
# flags.DEFINE_integer('encoder_depth', 5, 'network depth for U-Net encoder')
# flags.DEFINE_integer('final_channel_num', 64, 'final number of channels')
# flags.DEFINE_integer('start_channel_num', 64, 'start number of channels')
# decoder
# flags.DEFINE_integer('decoder_depth', 1, 'network depth for Gated_PixelCNN decoder')
# flags.DEFINE_integer('num_of_fms', 64, 'number of feature maps in convolutional layer')
# flags.DEFINE_integer('num_of_classes', 21, 'number of classes')

configure = flags.FLAGS

def main(_):
    model = Model(tf.Session(), configure)
    model.train()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    tf.app.run()