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

        self.train_able = cfg.TRAIN_ABLE

        self.learning_rate = cfg.LEARNING_RATE

        self.num_classes = len(cfg.CLASSES)

        self.loss_weights = cfg.LOSS_WEIGHTS

        self.model_path = cfg.MODEL_PATH

        self.batch_size = cfg.BATCH_SIZE

        self.target_size = cfg.TARGET_SIZE

        self.keep_rate = cfg.KEEP_RATE

        self.neg_weights = cfg.NEG_WEIGHTS

        self.reader = Reader(is_training=is_training)

        self.wd = 5e-4

        self.x = tf.placeholder(
            tf.float32, [None, self.target_size, self.target_size, 3])

        self.y0 = tf.placeholder(
            tf.int32, [None, 32, 32])
        self.y1 = tf.placeholder(
            tf.int32, [None, 64, 64])
        self.y2 = tf.placeholder(
            tf.int32, [None, 128, 128])
        self.y3 = tf.placeholder(
            tf.int32, [None, self.target_size, self.target_size])

        self.y = [self.y0, self.y1, self.y2, self.y3]

        self.y_hat = self.network(self.x)

        self.loss, self.regular_loss = self.sample_loss(self.y, self.y_hat)

        self.saver = tf.train.Saver()

    def sample_loss(self, labels, logits):

        losses = []

        for i in range(4):

            pos = labels[i] > 0
            pos_mask = tf.cast(pos, tf.float32)

            neg_mask = tf.ones(shape=tf.shape(pos_mask))
            neg_mask = tf.multiply(neg_mask, self.neg_weights)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels[i], logits=logits[i]
            )

            loss_weights = tf.where(pos, pos_mask, neg_mask)

            loss = tf.losses.compute_weighted_loss(loss, loss_weights)

            losses.append(loss)

        # losses = tf.reduce_mean(losses)

        losses = tf.losses.compute_weighted_loss(
            losses, self.loss_weights
        )

        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)

        regular_loss = tf.add_n(regularization_losses)

        return losses, regular_loss

    def add_weight_decay(self, var):

        weight_decay = tf.multiply(
            tf.nn.l2_loss(var), self.wd, name='weight_loss')

        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             weight_decay)
        return var

    def network(self, inputs, scope='ssd_512_vgg'):

        num_classes = self.num_classes
        train = self.is_training

        with tf.variable_scope(scope, 'ssd_512_vgg', [inputs], reuse=None):

            # Block 1
            net = slim.repeat(inputs, 2, slim.conv2d,
                              64, [3, 3], scope='conv1', trainable=self.train_able)
            net = slim.max_pool2d(net, [2, 2], scope='pool1', padding='SAME')

            # Block 2
            net = slim.repeat(net, 2, slim.conv2d,
                              128, [3, 3], scope='conv2', trainable=self.train_able)
            net = slim.max_pool2d(net, [2, 2], scope='pool2', padding='SAME')
            pool2 = net

            # Block 3
            net = slim.repeat(net, 3, slim.conv2d,
                              256, [3, 3], scope='conv3', trainable=self.train_able)
            net = slim.max_pool2d(net, [2, 2], scope='pool3', padding='SAME')
            pool3 = net

            # Block 4
            net = slim.repeat(net, 3, slim.conv2d,
                              512, [3, 3], scope='conv4', trainable=self.train_able)

            net = slim.max_pool2d(net, [2, 2], scope='pool4', padding='SAME')
            pool4 = net

            # Block 5
            net = slim.repeat(net, 3, slim.conv2d,
                              512, [3, 3], scope='conv5', trainable=self.train_able)

            # Block 6
            net = slim.conv2d(net, 1024, [3, 3],
                              2, scope='conv6', trainable=self.train_able)

            # Block 7
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')

            # up_pool 1
            net = self._unpool_layer(net,
                                     shape=tf.shape(pool4),
                                     num_classes=num_classes,
                                     name='up_pool1',
                                     ksize=4, stride=2)
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

            weights = self.get_deconv_filter(f_shape)

            self.add_weight_decay(weights)

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

    def run_test(self, num=4):

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(cfg.MODEL_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for _ in range(num):

                value, raw = self.reader.generate(1)

                images = value['images']
                labels = value['labels']

                feed_dict = {self.x: images}

                v = sess.run(self.y_hat, feed_dict)

                a1 = np.squeeze(np.argmax(v[-1], axis=-1))
                a2 = np.squeeze(labels[-1])

                tmp = np.hstack((a1, a2))

                plt.imshow(tmp)
                plt.show()

                raw = np.hstack([raw, raw])

                result = self.reader.decode_label(tmp, raw)

                plt.imshow(result)
                plt.show()

    def train_net(self):

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # self.optimizer = tf.compat.v1.train.MomentumOptimizer(
        #     learning_rate=self.learning_rate, momentum=0.9)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        self.train_step = self.optimizer.minimize(self.loss)

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(cfg.MODEL_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for i in range(cfg.EPOCHES):
                loss_list = []
                regular_loss_list = []
                for batch in range(cfg.BATCHES):

                    value, _ = self.reader.generate(self.batch_size)

                    images = value['images']
                    labels = value['labels']

                    feed_dict = {self.x: images,
                                 self.y0: labels[0],
                                 self.y1: labels[1],
                                 self.y2: labels[2],
                                 self.y3: labels[3]}

                    _, loss, regular_loss = sess.run(
                        [self.train_step, self.loss, self.regular_loss], feed_dict)

                    loss_list.append(loss)
                    regular_loss_list.append(regular_loss)

                    # print('batch:{} loss:{}'.format(batch, loss), end='\r')

                loss_values = np.array(loss_list)
                loss_values = np.mean(loss_values)

                regular_loss_values = np.array(regular_loss_list)
                regular_loss_values = np.mean(regular_loss_values)

                with open('./result.txt', 'a') as f:
                    f.write(str(loss_values)+'\n')

                self.saver.save(sess, os.path.join(
                    cfg.MODEL_PATH, 'model.ckpt'))

                print('epoch:{} loss:{} regular_loss:{}'.format(
                    self.reader.epoch, loss_values, regular_loss_values))


if __name__ == "__main__":

    net = Net(is_training=True)

    net.run_test(16)

    net.train_net()
