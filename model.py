import os
import sys
import tensorflow as tf
import numpy as np
import time

# from tensorflow.examples.tutorials.mnist import input_data
from binarized_mnist import DataLoader
from VAE import VAE

class Model():
    
    def __init__(self, sess, config):
        self.sess = sess
        self.conf = config
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)
        if not os.path.exists(self.conf.logdir):
            os.makedirs(self.conf.logdir)
        self.train_data = DataLoader(self.conf.data_dir, 'train', self.conf.batch_size)
        self.test_data = DataLoader(self.conf.data_dir, 'test', self.conf.batch_size)
        self.configure_network()

    def configure_network(self):
        self.build_network()
        self.train_summary = self.configure_summary('train')
        self.valid_summary = self.configure_summary('valid')
        # self.rng = np.random.RandomState(self.conf.random_seed)
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
            
    def build_network(self):
        self.X = tf.placeholder(tf.float32, [self.conf.batch_size, self.conf.height, self.conf.width, self.conf.channel])
        model = VAE(self.X, self.conf)
        # self.tsample = model.get_tsample()
        self.kl_loss, self.ce_loss, loss = model.get_loss()
        self.train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', update_ops=[])
        self.gsample = model.get_gsample()
        self.log_marginal_likelihood_estimate = model.log_marginal_likelihood_estimate()

    def configure_summary(self, name):
        summarys = []
        if name == 'train':
            summarys.append(tf.summary.scalar(name+'/kl_loss', self.kl_loss))
            summarys.append(tf.summary.scalar(name+'/l2_loss', self.l2_loss))
        # if name == 'train':
        #     summarys.append(tf.summary.image(name+'/train_input', self.X, max_outputs = 20))
        #     summarys.append(tf.summary.image(name+'/train_output', self.tsample, max_outputs = 20))
        if name == 'valid':
            summarys.append(tf.summary.image(name+'/prediction', self.gsample, 20))
        summary = tf.summary.merge(summarys)
        return summary

    def train(self):
        for epoch in range(1, self.conf.max_epochs+1):
            begin = time.time()
            
            if epoch == 1:
                if self.conf.reload_epoch > 0:
                    self.reload(self.conf.reload_epoch)

            # train for one epoch (for binarized data)
            train_kl_losses = []
            train_ce_losses = []
            for d in self.train_data:
                X = np.reshape(d, (self.conf.batch_size, self.conf.height, self.conf.width, self.conf.channel))
                feed_dict = {self.X: X}
                kl_loss, ce_loss, _ = self.sess.run([self.kl_loss, self.ce_loss, self.train_op], feed_dict=feed_dict)
                train_kl_losses.append(kl_loss)
                train_ce_losses.append(ce_loss)
            summary = self.sess.run(self.valid_summary, feed_dict=feed_dict)
            self.save_summary(summary, epoch, 'valid')
            train_kl_loss = np.mean(train_kl_losses)
            train_ce_loss = np.mean(train_ce_losses)
            print("Epoch %d, time = %ds, train kl loss = %.4f, train ce loss = %.4f" % (
                epoch, time.time() - begin, train_kl_loss, train_ce_loss))
            sys.stdout.flush()

            # after one epoch, do test: calculate log_marginal_likelihood_estimate (for binarized data)
            sum_ll = []
            for d in self.test_data:
                X = np.reshape(d, (self.conf.batch_size, self.conf.height, self.conf.width, self.conf.channel))
                feed_dict = {self.X: X}
                sample_ll = []
                for j in range(1000):
                    sample_ll.append(self.sess.run(self.log_marginal_likelihood_estimate, feed_dict=feed_dict))
                sample_ll = np.array(sample_ll).transpose((1,0))
                # print(sample_ll.shape)
                m = np.amax(sample_ll, axis=1, keepdims=True)
                log_marginal_estimate = m + np.log(np.mean(np.exp(sample_ll - m), axis=1, keepdims=True))
                sum_ll.append(np.mean(log_marginal_estimate))
            sum_ll = np.mean(sum_ll)
            print("---nll: %d" % sum_ll)
            sys.stdout.flush()

            if epoch % self.conf.save_interval == 0:
                self.save(epoch)


    def save_summary(self, summary, step, name):
        print('---->summarizing %s %d' % (name,step))
        self.writer.add_summary(summary, step)

    def save(self, epoch):
        print('---->saving', epoch)
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=epoch)

    def reload(self, epoch):
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(epoch)
        if not os.path.exists(model_path + '.meta'):
            print('------- no such checkpoint')
            return
        self.saver.restore(self.sess, model_path)
        print('Model restored.')