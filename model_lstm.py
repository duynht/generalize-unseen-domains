import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import lrelu
from tensorflow.contrib import rnn
from tf.keras.layers import Dense


class Model(object):
    """Tensorflow model
    """

    def __init__(self, mode='train'):

        self.n_classes = 2
        self.img_size = 32
        self.n_steps = 500
        self.n_input = 90
        self.no_channels = 1
        self.n_hidden = 200

    def encoder(self, images, reuse=False, return_feat=False):
        return self.encoder_lstm(images, reuse, return_feat)

    def encoder_lstm(self, images, reuse=False, return_feat=False):

        weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        with tf.variable_scope('encoder', reuse=reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
                with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, padding='VALID'):

                    # Prepare data shape to match `rnn` function requirements
                    # Current data input shape: (batch_size, n_steps, n_input)
                    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

                    # Permuting batch_size and n_steps
                    net = tf.transpose(images, [1, 0, 2])

                    # Reshaping to (n_steps*batch_size, n_input)
                    net = tf.reshape(net, [-1, self.n_input])

                    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
                    net = tf.split(net, self.n_steps, 0)

                    # Define a lstm cell with tensorflow
                    lstm_cell = rnn.BasicLSTMCell(
                        self.n_hidden, forget_bias=1.0)

                    # Get lstm cell output
                    outputs, states = rnn.static_rnn(
                        lstm_cell, net, dtype=tf.float32)

                    # Linear activation, using rnn inner loop last output
                    return slim.fully_connected(net, self.no_classes, activation_fn=None, scope='fco')
                    # return tf.matmul(outputs[-1], weights['out']) + biases['out']

    def build_model(self):

        # images placeholder
        self.z = tf.placeholder(
            tf.float32, [None, self.n_steps, self.n_input, self.no_channels], 'z')
        # labels placeholder
        self.labels = tf.placeholder(tf.int64, [None], 'labels')

        # images-for-gradient-ascent variable
        self.z_hat = tf.get_variable(
            'z_hat', [self.batch_size, self.n_steps, self.n_input, self.no_channels])
        # op to assign the value fed to self.z to the variable
        self.z_hat_assign_op = self.z_hat.assign(self.z)

        self.logits = self.encoder(self.z)
        self.logits_hat = self.encoder(self.z_hat, reuse=True)

        # for evaluation
        self.pred = tf.argmax(self.logits, 1)
        self.correct_pred = tf.equal(self.pred, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # variables for the minimizer are the net weights, variables for the maxmizer are the images' pixels
        t_vars = tf.trainable_variables()
        min_vars = [var for var in t_vars if 'z_hat' not in var.name]
        max_vars = [var for var in t_vars if 'z_hat' in var.name]

        # loss for the minimizer
        self.min_loss = slim.losses.sparse_softmax_cross_entropy(
            self.logits, self.labels)

        # first term of the loss for the maximizer (== loss for the minimizer)
        self.max_loss_1 = slim.losses.sparse_softmax_cross_entropy(
            self.logits_hat, self.labels)

        # second term of the loss for the maximizer
        self.max_loss_2 = slim.losses.mean_squared_error(self.encoder(
            self.z, reuse=True, return_feat=True), self.encoder(self.z_hat, reuse=True, return_feat=True))

        # final loss for the maximizer
        self.max_loss = self.max_loss_1 - self.gamma * self.max_loss_2

        # we use Adam for the minimizer and vanilla gradient ascent for the maximizer
        self.min_optimizer = tf.train.AdamOptimizer(self.learning_rate_min)
        self.max_optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate_max)

        # minimizer
        self.min_train_op = slim.learning.create_train_op(
            self.min_loss, self.min_optimizer, variables_to_train=min_vars)
        # maximizer (-)
        self.max_train_op = slim.learning.create_train_op(
            -self.max_loss, self.max_optimizer, variables_to_train=max_vars)

        min_loss_summary = tf.summary.scalar('min_loss', self.min_loss)
        max_loss_summary = tf.summary.scalar('max_loss', self.max_loss)

        accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
        self.summary_op = tf.summary.merge(
            [min_loss_summary, max_loss_summary, accuracy_summary])
