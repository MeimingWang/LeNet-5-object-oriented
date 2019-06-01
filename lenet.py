import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np

class Lenet(object):
    def __init__(self, mu, sigma, learning_rate, batch_size):
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self._build_graph()

    def _build_graph(self, network_name = 'Lenet'):
        self._setup_placeholders_graph()
        self._build_network_graph(network_name)
        self._compute_loss_graph()
        self._compute_acc_graph()
        self._creat_train_op_graph()
        self.merged_summary = tf.summary.merge_all()

    def _setup_placeholders_graph(self):
        self.raw_input_image = tf.placeholder('float', shape = [None, 784], name = 'raw_input_image')
        self.raw_input_label = tf.placeholder('float', shape = [None, 10], name = 'raw_input_label')          # real_label

    def _cnn_layer(self, scope_name, W_name, b_name, x, filter_shape, conv_strides, padding_tag = 'VALID'):
        with tf.variable_scope(scope_name):
            weight = tf.Variable(tf.truncated_normal(shape = filter_shape, mean = self.mu, stddev = self.sigma),
                                 name = W_name)
            bias = tf.Variable(tf.zeros(filter_shape[3]), name = b_name)
            conv = tf.nn.conv2d(x, weight, strides = conv_strides, padding = padding_tag) + bias
            return conv

    def _pooling_layer(self, scope_name, x, pool_ksize, pool_strides, padding_tag = 'VALID'):
        with tf.variable_scope(scope_name):
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize = pool_ksize, strides = pool_strides, padding = padding_tag)
            return x

    def _fully_connected_layer(self, scope_name, W_name, b_name, x, W_shape):
        with tf.variable_scope(scope_name):
            fc_W = tf.Variable(tf.truncated_normal(shape = W_shape, mean = self.mu, stddev = self.sigma),
                               name = W_name)
            fc_b = tf.Variable(tf.zeros(W_shape[1]), name = b_name)
            fc = tf.matmul(x, fc_W) + fc_b
            return fc
    def _build_network_graph(self, scope_name):
        with tf.variable_scope(scope_name):

            #padding
            self.input_x = tf.reshape(self.raw_input_image, [-1, 28, 28, 1])
            self.input_pad = tf.pad(self.input_x, paddings = [[0,0],[2,2],[2,2],[0,0]], mode = 'CONSTANT')

            # Layer 1: Input:N*32*32*1, Output:28*28*6
            conv1 = self._cnn_layer('layer_1_conv', 'conv1_w', 'conv1_b', self.input_pad, [5,5,1,6], [1,1,1,1])
            self.conv1 = tf.nn.relu(conv1)
            self.pool1 = self._pooling_layer('layer_1_pooling', self.conv1, [1,2,2,1], [1,2,2,1])

            #Layer 2: Input:N*14*14*6, Output:10*10*16
            conv2 = self._cnn_layer('layer_2_conv', 'conv2_w', 'conv2_b', self.pool1, [5,5,6,16], [1,1,1,1])
            self.conv2 = tf.nn.relu(conv2)
            self.pool2 = self._pooling_layer('layer_2_pooling', self.conv2, [1,2,2,1], [1,2,2,1])

            #Flatten Layer: Input;N*5*5*16, Output: N*400
            fc0 = tf.contrib.layers.flatten(self.pool2)

            # Fc Layer 1: Input:N*400, Output:N*120
            fc1 = self._fully_connected_layer('layer_3_fc', 'fc1_w', 'fc1_b', fc0, [400,120])
            self.fc1 = tf.nn.relu(fc1)

            # Fc Layer 2: Input:N*120, Output:N*84
            fc2 = self._fully_connected_layer('layer_4_fc', 'fc2_w', 'fc2_b', self.fc1, [120,84])
            self.fc2 = tf.nn.relu(fc2)

            # Fc Layer 3: Input:N*84, Output:N*10
            self.logits = self._fully_connected_layer('layer_5_fc', 'fc3_w', 'fc3_b', self.fc2, [84,10])
            self.y_predicted = tf.nn.softmax(self.logits)
            tf.summary.histogram('y_predicted', self.y_predicted)

    def _compute_loss_graph(self):
        with tf.name_scope('loss_function'):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.raw_input_label)
            self.loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss', self.loss)

    def _compute_acc_graph(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.raw_input_label, 1))
            self.accuracy =tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

    def _creat_train_op_graph(self):
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.training_step =self.optimizer.minimize(self.loss)


