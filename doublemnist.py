#!/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib.layers import flatten, linear, fully_connected, conv2d, batch_norm
from tensorflow.contrib.layers.python.layers import utils
from sacred import Experiment, Ingredient
from task import task_ingredient, Task
import numpy as np
import os
from utils import *

import numpy as np
na = np.newaxis

ex = Experiment('doublemnist', ingredients=[task_ingredient])

img_h, img_w = 96, 96

@task_ingredient.config
def task_config():
    batch_size = 128
    learning_rate = 0.01
    drop1 = 30
    drop2 = 60
    end_epoch = 80
    keep_prob = 0.5
    optimizer = 'momentum'
    name = 'doublemnist'
    even_labels = True

@ex.config
def cfg():
    dropout = 'information'
    activations = 'relu'
    beta = 0.5
    max_alpha = 1.0
    lognorm_prior = False
    weight_decay = 0.0
    filter_percentage = 1.0
    alpha_mode = 'information'

@ex.named_config
def softplus():
    activations = 'softplus'
    lognorm_prior = True

class MyTask(Task):

    def __init__(self):
        train = np.load('datasets/doublemnist-train.npz')
        test = np.load('datasets/doublemnist-test.npz')
        self.dataset = {'train': (train['data'], train['labels']),
                        'test': (test['data'], test['labels'])}
        self.dataset['valid'] = self.dataset['test']
        print("Double MNIST dataset loaded.")

    @task_ingredient.capture
    def build_placeholders(self, batch_size):
        '''Creates the placeholders for this model'''
        self.keep_prob = tf.placeholder(tf.float32, shape=[])
        self.initial_keep_prob = tf.placeholder(tf.float32, shape=[])
        self.sigma0 = tf.placeholder(tf.float32, shape=[])
        self.x = tf.placeholder(tf.float32, shape=[batch_size, img_h, img_w,1])  # input (batch_size * x_size)
        self.y = tf.placeholder(tf.float32, shape=[batch_size, 5]) 
        self.is_training = tf.placeholder(tf.bool, shape=[])

    @ex.capture
    def conv(self, inputs, num_outputs, activations, normalizer_fn = batch_norm, kernel_size=3, stride=1, scope=None):
        '''Creates a convolutional layer with default arguments'''
        if activations == 'relu':
            activation_fn = tf.nn.relu
        elif activations == 'softplus':
            activation_fn = tf.nn.softplus
        else:
            raise ValueError("Invalid activation function.")
        return conv2d( inputs = inputs,
            num_outputs = num_outputs,
            kernel_size = kernel_size,
            stride = stride,
            padding = 'SAME',
            activation_fn = activation_fn,
            normalizer_fn = batch_norm,
            scope=scope )

    @ex.capture
    def information_pool(self, inputs, max_alpha, alpha_mode, lognorm_prior, num_outputs=None, stride=2, scope=None):
        if num_outputs is None:
            num_ouputs = inputs.get_shape()[-1]
        # Creates the output convolutional layer
        network = self.conv(inputs, num_outputs=int(num_outputs), stride=stride)
        with tf.variable_scope(scope,'information_dropout'):
            # Computes the noise parameter alpha for the output
            alpha = conv2d(inputs, num_outputs=int(num_outputs), kernel_size=3,
                stride=stride, activation_fn=tf.sigmoid, scope='alpha')
            # Rescale alpha in the allowed range and add a small value for numerical stability
            alpha = 0.001 + max_alpha * alpha
            # Computes the KL divergence using either log-uniform or log-normal prior
            if not lognorm_prior:
                kl = - tf.log(alpha/(max_alpha + 0.001))
            else:
                mu1 = tf.get_variable('mu1', [], initializer=tf.constant_initializer(0.))
                sigma1 = tf.get_variable('sigma1', [], initializer=tf.constant_initializer(1.))
                kl = KL_div2(tf.log(tf.maximum(network,1e-4)), alpha, mu1, sigma1)
            tf.add_to_collection('kl_terms', kl)
        # Samples the noise with the given parameter
        e = sample_lognormal(mean=tf.zeros_like(network), sigma = alpha, sigma0 = self.sigma0)
        # Gather all infodropout layer activations in a collection
        tf.add_to_collection('dropout_layer_activations', network)
        # Returns the noisy output of the dropout
        return network * e

    @ex.capture
    def conv_dropout(self, inputs, num_outputs, dropout):
        if dropout == 'information':
            network = self.information_pool(inputs, num_outputs=num_outputs)
        elif dropout == 'binary':
            network = self.conv(inputs, num_outputs, stride=2)
            network = tf.nn.dropout(network, self.keep_prob)
            tf.add_to_collection('dropout_layer_activations', network)
        elif dropout == 'none':
            network = self.conv(inputs, num_outputs, stride=2)
        else:
            raise ValueError("Invalid dropout value")
        return network

    @ex.capture
    def build_network(self, inputs, filter_percentage):
        network = inputs
        # 96x96
        network = self.conv(network, 32)
        network = self.conv(network, 32)
        network = self.conv_dropout(network, 32)
        # 48x48
        network = self.conv(network, 64)
        network = self.conv(network, 64)
        network = self.conv_dropout(network, 64)
        # 24x24
        network = self.conv(network, 96)
        network = self.conv(network, 96)
        network = self.conv_dropout(network, 96)
        # 12x12
        network = self.conv(network, 192)
        network = self.conv(network, 192)
        network = self.conv_dropout(network, 192)
        # 6x6
        network = self.conv(network, 192)
        network = self.conv(network, 192, kernel_size=1)
        network = self.conv(network, 5, kernel_size=1)
        network = spatial_global_mean(network)

        return network


    @ex.capture
    def build_loss(self, beta, task, weight_decay):
        batch_size = task['batch_size']
        with tf.variable_scope("network") as scope:
            network = self.build_network(self.x)
            logits = linear(network, num_outputs=5)
        with tf.name_scope('loss'):
            kl_terms = [batch_average(kl) for kl in tf.get_collection('kl_terms')]
            if not kl_terms:
                kl_terms = [tf.constant(0.)]
            N_train = self.dataset['train'][0].shape[0]
            Lz = tf.add_n(kl_terms)/N_train
            Lx = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.y))
            beta = tf.constant(beta)
            L2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() ])
            loss = Lx + beta * Lz + weight_decay * L2
            correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(self.y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.logits = logits
        self.loss = loss
        self.error = (1. - accuracy) * 100.
        self.Lx = Lx
        self.Lz = Lz
        self.beta = beta


    @task_ingredient.capture
    def preprocess_labels(self, label_batch, even_labels):
        label_converter = (np.arange(5)[:,na]*np.ones(2, dtype='int')[na,:]).flatten()
        if even_labels:
            labels = label_converter[label_batch[:, 0]]
        else:
            labels = label_converter[label_batch[:, 1]]
        labels = np.eye(5)[labels]
        return labels


    @task_ingredient.capture
    def train_batch(self, batch, stats, batch_size, keep_prob):
        xtrain, ytrain = batch
        xtrain = xtrain.reshape(-1,img_h,img_w,1)
        ytrain = self.preprocess_labels(ytrain)
        feed_dict = {self.x: xtrain, self.y: ytrain, self.sigma0: 1., self.keep_prob: keep_prob, self.learning_rate: self.current_learning_rate, self.is_training: True}
        batch_cost, batch_error, batch_Lx, batch_Lz, batch_beta, _ = self.sess.run( [ self.loss, self.error, self.Lx, self.Lz, self.beta, self.train_op], feed_dict)

        stats.push(Lx = batch_Lx)
        stats.push(Lz = batch_Lz)
        stats.push(train = batch_cost)
        stats.push(error = batch_error)

    # 
    @ex.capture
    def valid_batch(self, batch, stats):
        xtrain, ytrain = batch
        xtrain = xtrain.reshape(-1,img_h,img_w,1)
        ytrain = self.preprocess_labels(ytrain)
        feed_dict = {self.x: xtrain, self.y: ytrain, self.sigma0: 0., self.keep_prob: 1., self.is_training: False}
        batch_cost, batch_error = self.sess.run( [ self.loss, self.error], feed_dict)

        stats.push(train = batch_cost)
        stats.push(error = batch_error)

    @ex.capture
    def dry_run(self):
        '''
        Since the statistics learned by batch normalization with dropout are
        incorrect when dropout is disabled,
        we do a dry run without dropout in order to relearn them before testing.
        '''
        for batch in self.iterate_minibatches('train'):
            xtrain, ytrain = batch
            xtrain = xtrain.reshape(-1,img_h,img_w,1)
            ytrain = self.preprocess_labels(ytrain)
            feed_dict = {self.x: xtrain, self.y: ytrain, self.sigma0: 0., self.keep_prob: 1., self.is_training: True}
            batch_cost = self.sess.run( [ self.loss], feed_dict)

    @task_ingredient.capture
    def plot_kl(self, batch_size, even_labels):
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        # xtrain, ytrain = self.get_test_batch()
        xtrain, ytrain = [x[:batch_size] for x in self.dataset['valid']]
        xtrain = xtrain.reshape(batch_size,img_h,img_w,1)
        ytrain_p = self.preprocess_labels(ytrain)
        feed_dict = {self.x: xtrain, self.y: ytrain_p, self.sigma0: 0., self.keep_prob: 1.}
        logits, kls, acts = self.sess.run([self.logits, tf.get_collection('kl_terms'),
                             tf.get_collection('dropout_layer_activations')], feed_dict)
        raw_kls = kls
        kls = [k.sum(axis=-1) for k in kls]
        kls = [k - k.min() for k in kls]

        basepath = 'plots/'+self.get_name()+'/'
        if not os.path.exists(basepath):
            os.makedirs(basepath)

        rows = 1 + int(len(kls) > 0)
        cols = 1 + len(acts)

        for j in range(5):
            plt.clf()
            fig = plt.figure(figsize=(14, 5))

            plt.subplot(rows, cols, 1)
            plt.axis('off')
            plt.imshow(xtrain[j,:,:,0], cmap='gray', interpolation='none')
            if even_labels:
                predicted_class = 2 * np.argmax(logits[j])
                plt.title('target: {}\nprediction: {}'.format(ytrain[j][0],
                                                              predicted_class))
            else:
                predicted_class = 2 * np.argmax(logits[j]) + 1
                plt.title('target: {}\nprediction: {}'.format(ytrain[j][1],
                                                              predicted_class))
            for i,act in enumerate(acts):
                plt.subplot(rows, cols, 2 + i)
                plt.title('Activation')
                plt.imshow(act[j].mean(2), cmap='Blues', interpolation='none')
                plt.axis('off')
                plt.colorbar()
                if len(kls) > 0:
                    plt.subplot(rows, cols, 2 + i + cols)
                    plt.title('KL-Div')
                    plt.imshow(kls[i][j], cmap='Blues', interpolation='none')
                    plt.axis('off')
                    plt.colorbar()

            plt.savefig(basepath+'ex_{}_acts.png'.format(j), bbox_inches='tight')


        cmap = 'viridis'

        # plot last-layer acts + kl-divs per class
        rows = 1 + int(len(kls) > 0)
        cols = 1 + acts[3].shape[2]
        for j in range(5):
            plt.clf()
            fig = plt.figure(figsize=(14, 5))

            plt.subplot(rows, cols, 1)
            plt.axis('off')
            plt.imshow(xtrain[j,:,:,0], cmap='gray', interpolation='none')
            if even_labels:
                predicted_class = 2 * np.argmax(logits[j])
                plt.title('target: {}\nprediction: {}'.format(ytrain[j][0],
                                                              predicted_class))
            else:
                predicted_class = 2 * np.argmax(logits[j]) + 1
                plt.title('target: {}\nprediction: {}'.format(ytrain[j][1],
                                                              predicted_class))

            for c_ind, c in enumerate(2 * np.arange(5) + (1 - int(even_labels))):
                act = acts[3][j][:,:,c_ind]
                plt.subplot(rows, cols, 2 + c_ind)
                plt.imshow(act, cmap=cmap)
                plt.title(c)
                plt.axis('off')

                if kls:
                    kl = raw_kls[3][j][:,:,c_ind]
                    plt.subplot(rows, cols, 2 + cols + c_ind)
                    plt.imshow(kl, cmap=cmap)
                    plt.axis('off')

            plt.savefig(basepath+'ex_{}_per_class_last_layer_acts_n_kls.png'.format(j), bbox_inches='tight')


mytask = task = MyTask()

@ex.command
def train():
    task.initialize()
    task.train()

@ex.command
def test(load=True):
    if load:
        task.initialize(_load=True, _log_dir='valid/')
    print("Dry run...")
    task.dry_run()
    print("Validating...")
    task.valid()

@ex.command
def plot():
    task.initialize(_load=True, _log_dir='other/')
    task.plot_kl()

@ex.automain
def run():
    train()
    # task.valid(load=False)
