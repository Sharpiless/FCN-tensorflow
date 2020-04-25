import tensorflow as tf
import numpy as np
import config as cfg
from read_data import Reader
from fcn_vgg import Net
from random_crop import random_crop
import matplotlib.pyplot as plt
import cv2
import random


class FCN_detector(object):

    def __init__(self):

        self.size = cfg.TARGET_SIZE

        self.target_size = cfg.TARGET_SIZE

        self.CLASSES = cfg.CLASSES

        self.layers = cfg.LAYERS_SHAPE

        self.COLOR_MAP = cfg.COLOR_MAP

        self.pixel_means = cfg.PIXEL_MEANS

        self.net = Net(is_training=False)

    def read_image(self, path):
    
        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float)

    def resize_image(self, image, label_image, with_label=True):
    
        image_shape = image.shape

        size_min = np.min(image_shape[:2])
        size_max = np.max(image_shape[:2])

        size = self.target_size + random.randint(0, 64)

        scale = float(size) / float(size_min)

        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)

        if with_label:

            label_image = cv2.resize(
                label_image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        return image, label_image

    def pre_process_image(self, path):

        image = self.read_image(path)

        image, _ = self.resize_image(
            image, None, with_label=False)

        image = random_crop(image, with_label=False)

        raw = image.copy()

        image = self.standardize(image)

        image = np.expand_dims(image, axis=0)

        return image, raw

    def standardize(self, image):
    
        mean = np.mean(image)
        var = np.mean(np.square(image-mean))

        image = (image - mean)/np.sqrt(var)

        return image
    
    def decode_label(self, label, raw):
    
        h, w = label.shape[:2]

        label_image = np.zeros((h*w, 3), dtype=np.int)

        label = label.reshape(-1)

        for i in range(1, len(self.CLASSES)):

            cls_index = np.where(label == i)

            color = self.COLOR_MAP[i]

            label_image[cls_index] = self.COLOR_MAP[i]

        result = label_image.reshape((h, w, 3))+raw

        return result.astype(np.int)

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

                result = sess.run(self.net.logits, feed_dict={
                                  self.net.x: image})

                label_image = np.squeeze(result)

                label_image = np.argmax(label_image, axis=-1)
                label_image = self.decode_label(label_image, raw)

                result = label_image

                results.append(result)

                plt.imshow(result)
                plt.show()

        return results

if __name__ == "__main__":
    
    de = FCN_detector()

    de.test('./demo.jpg')