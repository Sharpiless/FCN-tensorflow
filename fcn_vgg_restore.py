import numpy as np
import tensorflow as tf
import config as cfg
from read_data import Reader
from math import ceil
import os
from random_crop import random_crop
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import xavier_initializer
import cv2

slim = tf.contrib.slim


class Net(object):

    def __init__(self, is_training):

        self.is_training = is_training

        self.epoches = cfg.EPOCHES

        self.learning_rate = 0

        self.num_classes = len(cfg.CLASSES)

        self.model_path = cfg.MODEL_PATH

        self.batch_size = cfg.BATCH_SIZE

        self.target_size = cfg.TARGET_SIZE

        self.keep_rate = cfg.KEEP_RATE

        self.reader = Reader(is_training=is_training)

        self.x = tf.placeholder(
            tf.float32, [None, self.target_size, self.target_size, 3])

        self.y_hat = self.network(self.x)

        with open('var.txt', 'r') as f:
            variables = tf.contrib.framework.get_variables_to_restore()
            var_s = f.read().splitlines()
            variables_to_restore = [
                v for v in variables if v.name in var_s]

        self.saver = tf.train.Saver(variables_to_restore)

    def network(self, inputs, scope='ssd_512_vgg'):

        num_classes = self.num_classes
        train = self.is_training

        with tf.variable_scope(scope, 'ssd_512_vgg', [inputs], reuse=None):

            # Block 1
            net = slim.repeat(inputs, 2, slim.conv2d,
                              64, [3, 3], scope='conv1', trainable=False)
            net = slim.max_pool2d(net, [2, 2], scope='pool1', padding='SAME')

            # Block 2
            net = slim.repeat(net, 2, slim.conv2d,
                              128, [3, 3], scope='conv2', trainable=False)
            net = slim.max_pool2d(net, [2, 2], scope='pool2', padding='SAME')
            pool2 = net

            # Block 3
            net = slim.repeat(net, 3, slim.conv2d,
                              256, [3, 3], scope='conv3', trainable=False)
            net = slim.max_pool2d(net, [2, 2], scope='pool3', padding='SAME')
            pool3 = net

            # Block 4
            net = slim.repeat(net, 3, slim.conv2d,
                              512, [3, 3], scope='conv4', trainable=False)

            net = slim.max_pool2d(net, [2, 2], scope='pool4', padding='SAME')
            pool4 = net

            # Block 5
            net = slim.repeat(net, 3, slim.conv2d,
                              512, [3, 3], scope='conv5', trainable=False)

            # Block 6
            net = slim.conv2d(net, 1024, [3, 3],
                              2, scope='conv6', trainable=False)

            # Block 7
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')

            # up_pool 1
            net = self._unpool_layer(net,
                                     shape=tf.shape(pool4),
                                     num_classes=num_classes,
                                     name='up_pool1',
                                     ksize=4, stride=2)
            self.test = tf.nn.softmax(net, axis=-1)
            score_pool1 = slim.conv2d(
                pool4, num_classes, [1, 1], 1, scope='score_pool1')
            net = tf.add(net, score_pool1)
            up_pool1 = tf.nn.softmax(net, axis=-1)

            # up_pool 2
            net = self._unpool_layer(net,
                                     shape=tf.shape(pool3),
                                     num_classes=num_classes,
                                     name='up_pool2',
                                     ksize=4, stride=2)
            score_pool2 = slim.conv2d(
                pool3, num_classes, [1, 1], 1, scope='score_pool2')
            net = tf.add(net, score_pool2)
            up_pool2 = tf.nn.softmax(net, axis=-1)

            # up_pool 3
            net = self._unpool_layer(net,
                                     shape=tf.shape(pool2),
                                     num_classes=num_classes,
                                     name='up_pool3',
                                     ksize=4, stride=2)
            score_pool3 = slim.conv2d(
                pool2, num_classes, [1, 1], 1, scope='score_pool3')
            net = tf.add(net, score_pool3)
            up_pool3 = tf.nn.softmax(net, axis=-1)

            # up_pool 4
            logits = self._unpool_layer(net,
                                        shape=tf.shape(inputs),
                                        num_classes=num_classes,
                                        name='up_pool4',
                                        ksize=8, stride=4)
            logits = tf.nn.softmax(logits, axis=-1)

            return up_pool1, up_pool2, up_pool3, logits

    def _unpool_layer(self, bottom, shape,
                      num_classes, name,
                      ksize=4, stride=2):

        strides = [1, stride, stride, 1]

        with tf.variable_scope(name):

            in_features = bottom.get_shape()[3].value

            new_shape = [shape[0], shape[1], shape[2], num_classes]

            output_shape = tf.stack(new_shape)

            f_shape = [ksize, ksize, num_classes, in_features]

            '''weights = tf.get_variable(
                'W', f_shape, tf.float32, xavier_initializer())'''
            weights = self.get_deconv_filter(f_shape)

            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

        return deconv

    def get_deconv_filter(self, f_shape):
        # 双线性插值
        width = f_shape[0]
        height = f_shape[1]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights+0.1,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def run_test(self):

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(cfg.MODEL_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            value = self.reader.generate(1)

            images = value['images']
            labels = value['labels']

            feed_dict = {self.x: images}

            v = sess.run(self.y_hat, feed_dict)

            for i in range(4):

                a1 = np.squeeze(np.argmax(v[i], axis=-1))
                a2 = np.squeeze(labels[i])

                tmp = np.hstack((a1, a2))

                plt.imshow(tmp)
                plt.show()

    def train_net(self):

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(cfg.MODEL_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            self.saver = tf.train.Saver()

            self.saver.save(sess, os.path.join(
                cfg.MODEL_PATH, 'model.ckpt'))


if __name__ == "__main__":

    net = Net(is_training=True)

    net.train_net()
