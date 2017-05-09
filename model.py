import os
import sys
import tensorflow as tf
import numpy as np
import time
from ops import parzen_cpu_batch

from tensorflow.examples.tutorials.mnist import input_data
from VAE import VAE

class Model():
    
    def __init__(self, sess, config):
        self.sess = sess
        self.conf = config
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)
        if not os.path.exists(self.conf.logdir):
            os.makedirs(self.conf.logdir)
        data = input_data.read_data_sets(self.conf.data_dir)
        self.train_data = data.train
        self.test_data = data.test
        self.valid_data = data.valid
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
        self.kl_loss, self.l2_loss, loss = model.get_loss()
        self.train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_or_create_global_step(), 
            learning_rate=self.conf.learning_rate, optimizer='Adam', update_ops=[])
        self.gsample = model.get_gsample()

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
            
            # train for one epoch
            train_kl_losses = []
            train_l2_losses = []
            for i in range(1, self.conf.updates_per_epoch+1):
                X, _ = self.train_data.next_batch(self.conf.batch_size)
                X = np.reshape(X, (self.conf.batch_size, self.conf.height, self.conf.width, self.conf.channel))
                feed_dict = {self.X: X}
                
                if i % self.conf.summary_step == 0:
                    kl_loss, l2_loss, _, summary = self.sess.run(
                        [self.kl_loss, self.l2_loss, self.train_op, self.train_summary], feed_dict=feed_dict)
                    self.save_summary(summary, self.conf.updates_per_epoch*(epoch-1)+i, 'train')
                else:
                    kl_loss, l2_loss, _ = self.sess.run(
                        [self.kl_loss, self.l2_loss, self.train_op], feed_dict=feed_dict)
                train_kl_losses.append(kl_loss)
                train_l2_losses.append(l2_loss)

                if i % self.conf.validation_interval == 0:
                    summary = self.sess.run(
                        self.valid_summary, feed_dict=feed_dict)
                    self.save_summary(summary, self.conf.updates_per_epoch*(epoch-1)+i, 'valid')
                
            train_kl_loss = np.mean(train_kl_losses)
            train_l2_loss = np.mean(train_l2_losses)
            print("Epoch %d, time = %ds, train kl loss = %.4f, train l2 loss = %.4f" % (
                epoch, time.time() - begin, train_kl_loss, train_l2_loss))
            sys.stdout.flush()

            if epoch % self.conf.save_interval == 0:
                self.save(epoch)

        # ---------begin test----------
                    
        # generate samples
        samples = []
        for i in range(100): # generate 100*100 samples
            samples.extend(self.sess.run(self.gsample))
        samples = np.array(samples)
        print (samples.shape)

        #choose
        sigmas = np.logspace(-1.0, 0.0, 10)
        lls = []
        for sigma in sigmas:
            # evaluation
            nlls = []
            for i in range(1,10+1): # number of valid batches = 10
                X, _ = self.valid_data.next_batch(self.conf.batch_size)
                nll = parzen_cpu_batch(X, samples, sigma=sigma, batch_size=self.conf.batch_size, num_of_samples=10000, data_size=784)
                nlls.extend(nll)
            
            nlls = np.array(nlls).reshape(1000) # 1000 valid images
            print("sigma: ", sigma)
            print("ll: %d" % (np.mean(nlls)))
            lls.append(np.mean(nlls))
        sigma = sigmas[np.argmax(lls)]
        
        # evaluation
        nlls = []
        for i in range(1,100+1): # number of test batches = 100
            X, _ = self.test_data.next_batch(self.conf.batch_size)
            nll = parzen_cpu_batch(X, samples, sigma=sigma, batch_size=self.conf.batch_size, num_of_samples=10000, data_size=784)
            nlls.extend(nll)
        
        nlls = np.array(nlls).reshape(10000) # 10000 test images
        print("sigma: ", sigma)
        print("ll: %d" % (np.mean(nlls)))
        print("se: %d" % (nlls.std() / np.sqrt(10000)))

        # ---------end test----------


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