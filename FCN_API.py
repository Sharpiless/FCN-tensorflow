import tensorflow as tf
import numpy as np
import config as cfg
from read_data import Reader
from fcn_resnet import Net
from random_crop import random_crop
import matplotlib.pyplot as plt


class FCN_detector(object):

    def __init__(self):

        self.size = cfg.TARGET_SIZE

        self.reader = Reader(is_training=False)

        self.net = Net(is_training=False)

    def pre_process_image(self, path):

        image = self.reader.read_image(path)

        image, _ = self.reader.resize_image(
            image, None, with_label=False)

        image = random_crop(image, with_label=False)

        raw = image.copy()

        image = self.reader.standardize(image)

        image = np.expand_dims(image, axis=0)

        return image, raw

    def test(self, paths):

        if isinstance(paths, str):
            paths = [paths]

        results = []

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(self.net.model_path)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.net.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for path in paths:

                image, raw = self.pre_process_image(path)

                result = sess.run(self.net.y_hat, feed_dict={
                                  self.net.x: image})

                label_image = np.squeeze(result)

                label_image = np.argmax(label_image, axis=-1)
                label_image = self.reader.decode_label(label_image)

                result = label_image

                results.append(result)

                plt.imshow(result)
                plt.show()

        return results

