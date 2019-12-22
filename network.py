import numpy as np
import tensorflow as tf
import config as cfg
from read_data import Reader
from math import ceil
import os
from random_crop import random_crop
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import xavier_initializer

slim = tf.contrib.slim


class Net(object):

    def __init__(self, is_training):

        self.is_training = is_training

        self.epoches = cfg.EPOCHES

        self.learning_rate = cfg.LEARNING_RATE

        self.num_classes = len(cfg.CLASSES)

        self.model_path = cfg.MODEL_PATH

        self.batch_size = cfg.BATCH_SIZE

        self.target_size = cfg.TARGET_SIZE

        self.keep_rate = cfg.KEEP_RATE

        self.reader = Reader(is_training=is_training)

        self.x = tf.placeholder(
            tf.float32, [None, self.target_size, self.target_size, 3])

        self.y = tf.placeholder(
            tf.int32, [None, self.target_size, self.target_size])

        self.y_hat = self.network(self.x)

        self.loss = self.sample_loss(self.y, self.y_hat)

        self.saver = tf.train.Saver()

    def sample_loss(self, labels, logits):

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )

        return tf.reduce_mean(loss)

    def network(self, inputs):

        num_classes = self.num_classes
        train = self.is_training

        with tf.variable_scope('vgg'):

            conv1_1 = self._conv_layer(inputs, 64, 'conv1_1')
            conv1_2 = self._conv_layer(conv1_1, 64, 'conv1_2')
            pool1 = self._max_pool(conv1_2, 'pool1')

            conv2_1 = self._conv_layer(pool1, 128, 'conv2_1')
            conv2_2 = self._conv_layer(conv2_1, 128, 'conv2_2')
            pool2 = self._max_pool(conv2_2, 'pool2')

            conv3_1 = self._conv_layer(pool2, 256, 'conv3_1')
            conv3_2 = self._conv_layer(conv3_1, 256, 'conv3_2')
            conv3_3 = self._conv_layer(conv3_2, 256, 'conv3_3')
            pool3 = self._max_pool(conv3_3, 'pool3')

            conv4_1 = self._conv_layer(pool3, 512, 'conv4_1')
            conv4_2 = self._conv_layer(conv4_1, 512, 'conv4_2')
            conv4_3 = self._conv_layer(conv4_2, 512, 'conv4_3')
            pool4 = self._max_pool(conv4_3, 'pool4')

            conv5_1 = self._conv_layer(pool4, 512, 'conv5_1')
            conv5_2 = self._conv_layer(conv5_1, 512, 'conv5_2')
            conv5_3 = self._conv_layer(conv5_2, 512, 'conv5_3')
            pool5 = self._max_pool(conv5_3, 'pool5')

        fc6 = self._conv_layer(pool5, 4096, k_size=1, name='fc6')

        if train:
            fc6 = tf.nn.dropout(fc6, self.keep_rate)

        fc7 = self._conv_layer(fc6, 4096, k_size=1, name='fc7')

        if train:
            fc7 = tf.nn.dropout(fc7, self.keep_rate)

        fc8 = self._conv_layer(
            fc7, 1000, k_size=1, name='fc8')

        # up pool 1
        unpool1 = self._unpool_layer(fc8,
                                     shape=tf.shape(pool4),
                                     num_classes=num_classes,
                                     name='unpool1',
                                     ksize=4, stride=2)
        score_pool4 = self._conv_layer(
            pool4, num_classes, k_size=1, name='score_pool4')
        fuse_pool4 = tf.add(unpool1, score_pool4)

        # up pool 2
        unpool2 = self._unpool_layer(fuse_pool4,
                                     shape=tf.shape(pool3),
                                     num_classes=num_classes,
                                     name='unpool2',
                                     ksize=4, stride=2)
        score_pool3 = self._conv_layer(
            pool3, num_classes, k_size=1, name='score_pool3')
        fuse_pool3 = tf.add(unpool2, score_pool3)

        # up pool 3
        unpool3 = self._unpool_layer(fuse_pool3,
                                     shape=tf.shape(pool2),
                                     num_classes=num_classes,
                                     name='unpool3',
                                     ksize=4, stride=2)
        score_pool2 = self._conv_layer(
            pool2, num_classes, k_size=1, name='score_pool2')
        fuse_pool2 = tf.add(unpool3, score_pool2)

        # up pool 4
        unpool4 = self._unpool_layer(fuse_pool2,
                                     shape=tf.shape(pool1),
                                     num_classes=num_classes,
                                     name='unpool4',
                                     ksize=4, stride=2)
        score_pool1 = self._conv_layer(
            pool1, num_classes, k_size=1, name='score_pool1')
        fuse_pool2 = tf.add(unpool4, score_pool1)

        # up pool 5
        logits = self._unpool_layer(fuse_pool2,
                                    shape=tf.shape(inputs),
                                    num_classes=num_classes,
                                    name='unpool5',
                                    ksize=4, stride=2)

        return tf.nn.softmax(logits, axis=-1)

    def _max_pool(self, bottom, name):

        pool = slim.max_pool2d(bottom, [2, 2], scope=name, padding='SAME')

        return pool

    def _conv_layer(self, bottom, filters, name, k_size=3):

        conv = slim.conv2d(bottom, filters, [
                           k_size, k_size], scope=name, padding='SAME')

        return conv

    def _unpool_layer(self, bottom, shape,
                      num_classes, name,
                      ksize=4, stride=2):

        strides = [1, stride, stride, 1]

        with tf.variable_scope(name):

            in_features = bottom.get_shape()[3].value

            new_shape = [shape[0], shape[1], shape[2], num_classes]

            output_shape = tf.stack(new_shape)

            f_shape = [ksize, ksize, num_classes, in_features]

            weights = tf.get_variable(
                'W', f_shape, tf.float32, xavier_initializer())

            deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                            strides=strides, padding='SAME')

        return deconv

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
                for batch in range(cfg.BATCHES):

                    value = self.reader.generate(self.batch_size)

                    images = value['images']
                    labels = value['labels']

                    feed_dict = {self.x: images,
                                 self.y: labels}

                    _, loss, pred = sess.run([self.train_step, self.loss, self.y_hat], feed_dict)

                    loss_list.append(loss)

                    print('batch:{} loss:{}'.format(batch, loss), end='\r')

                loss_values = np.array(loss_list)  # (64, 3)

                loss_values = np.mean(loss_values)

                with open('./result.txt', 'a') as f:
                    f.write(str(loss_values)+'\n')

                self.saver.save(sess, os.path.join(
                    cfg.MODEL_PATH, 'model.ckpt'))

                print('epoch:{} loss:{}'.format(
                    self.reader.epoch, loss_values))

    def test(self, image_path):

        if not isinstance(image_path, list):
            image_path = [image_path]

        self.is_training = False

        test = []

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(cfg.MODEL_PATH)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for path in image_path:

                image = self.reader.read_image(path)
                image, _ = self.reader.resize_image(
                    image, label_image=None, with_label=False)
                image = random_crop(image, None, with_label=False)

                image = np.expand_dims(image, axis=0)

                label_image = sess.run(self.y_hat, feed_dict={
                                       self.x: image-self.reader.pixel_means})
                label_image = np.squeeze(label_image)

                label_image = np.argmax(label_image, axis=-1)
                label_image = self.reader.decode_label(label_image)

                # result = np.vstack([image, label_image])
                result = label_image

                plt.imshow(result)
                plt.show()

                test.append(result)

        return test


if __name__ == "__main__":

    net = Net(is_training=True)

    net.train_net()

    test_path = ['./0.jpg', './1.jpg']

    test = net.test(test_path)

    plt.imshow(np.vstack(test))
    plt.show()
