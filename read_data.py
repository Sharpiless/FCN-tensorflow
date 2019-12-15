import config as cfg
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import os
import pickle
import random
from random_crop import random_crop


class Reader(object):
    def __init__(self, is_training):

        self.data_path = cfg.DATA_PATH

        self.is_training = is_training

        self.target_size = cfg.TARGET_SIZE

        self.CLASSES = cfg.CLASSES

        self.COLOR_MAP = cfg.COLOR_MAP

        self.pixel_means = cfg.PIXEL_MEANS

        self.cursor = 0

        self.epoch = 1

        self.pre_process()

    def read_image(self, path):

        image = cv2.imread(path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float)

    def load_one_info(self, name):

        image_path = os.path.join(self.data_path, 'JPEGImages', name+'.jpg')
        label_path = os.path.join(
            self.data_path, 'SegmentationClass', name+'.png')

        return {'image_path': image_path, 'label_path': label_path}

    def load_labels(self):

        is_training = 'train' if self.is_training else 'test'

        if not os.path.exists('./dataset'):
            os.makedirs('./dataset')

        pkl_file = os.path.join('./dataset', is_training+'_labels.pkl')

        if os.path.isfile(pkl_file):

            print('Load Label From '+str(pkl_file))
            with open(pkl_file, 'rb') as f:
                labels = pickle.load(f)

            return labels

        # else

        print('Load labels from: '+str(cfg.ImageSets_PATH))

        if self.is_training:
            txt_path = os.path.join(
                cfg.ImageSets_PATH, 'Segmentation', 'trainval.txt')
            # 这是用来存放训练集和测试集的列表的txt文件
        else:
            txt_path = os.path.join(
                cfg.ImageSets_PATH, 'Segmentation', 'val.txt')

        with open(txt_path, 'r') as f:
            self.image_name = [x.strip() for x in f.readlines()]

        labels = []

        for name in self.image_name:

            true_label = self.load_one_info(name)
            labels.append(true_label)

        with open(pkl_file, 'wb') as f:
            pickle.dump(labels, f)

        print('Successfully saving '+is_training+'data to '+pkl_file)

        return labels

    def resize_image(self, image, label_image, with_label=True):

        image_shape = image.shape

        size_min = np.min(image_shape[:2])
        size_max = np.max(image_shape[:2])

        size = self.target_size + random.randint(0, 64)

        scale = float(size) / float(size_min)

        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)

        if with_label:

            label_image = cv2.resize(
                label_image, dsize=(0, 0), fx=scale, fy=scale)

        return image, label_image

    def sample_weight(self, labels):

        shape = labels.shape
        labels = labels.reshape(-1)

        weight = np.zeros(shape=labels.shape)

        pos_index = np.where(labels != 0)[0]
        pos_num = pos_index.shape[0]

        neg_index = np.where(labels == 0)[0]
        neg_num = neg_index.shape[0]

        neg_num = min(neg_num, pos_num*3)
        neg_index = np.random.choice(neg_index, neg_num, replace=False)

        keep_index = np.hstack([pos_index, neg_index])

        weight[keep_index] = 1.

        return np.squeeze(np.reshape(weight, shape))

    def pre_process(self):

        true_labels = self.load_labels()

        if self.is_training:
            np.random.shuffle(true_labels)

        self.true_labels = true_labels

    def generate(self, batch_size):

        images = []
        labels = []
        weights = []

        for _ in range(batch_size):

            image_path = self.true_labels[self.cursor]['image_path']
            image = self.read_image(image_path)

            label_path = self.true_labels[self.cursor]['label_path']
            label_image = self.read_image(label_path)

            image, label_image = self.resize_image(image, label_image)
            image, label_image = random_crop(image, label_image)

            self.cursor += 1

            if self.cursor >= len(self.true_labels):
                np.random.shuffle(self.true_labels)

                self.cursor = 0
                self.epoch += 1

            label = self.fast_encode_label(label_image)
            weight = self.sample_weight(label)
            image -= self.pixel_means

            images.append(image)
            labels.append(label)
            weights.append(weight)

        images = np.stack(images)
        labels = np.stack(labels)
        weights = np.stack(weights)

        value = {'images': images, 'labels': labels, 'weights': weights}

        return value

    def encode_label(self, label_image):

        h, w = label_image.shape[:2]

        label = np.zeros(shape=(h, w, 1))

        for cls_, color in enumerate(self.COLOR_MAP):

            if cls_ == 0:
                continue

            for i in range(h):
                for j in range(w):

                    if all(label_image[i, j, :] == color):
                        label[i, j] = cls_

        return label

    def fast_encode_label(self, label_image):

        h, w = label_image.shape[:2]

        label = np.zeros(shape=(h*w, 1))

        label_image = label_image.reshape((-1, 3))

        r_image = label_image[:, 0]
        g_image = label_image[:, 1]
        b_image = label_image[:, 2]

        for cls_, color in enumerate(self.COLOR_MAP):

            if cls_ == 0:
                continue

            r, g, b = color

            cls_index = np.logical_and(r_image == r, g_image == g)
            cls_index = np.logical_and(cls_index, b_image == b)
            cls_index = np.where(cls_index)

            label[cls_index] = cls_

        return label.reshape((h, w))

    def decode_label(self, label):

        h, w = label.shape[:2]

        label_image = np.zeros((h*w, 3), dtype=np.int)

        label = label.reshape(-1)

        for i in range(1, len(self.CLASSES)):

            cls_index = np.where(label == i)

            color = self.COLOR_MAP[i]

            label_image[cls_index] = self.COLOR_MAP[i]

        return label_image.reshape((h, w, 3))


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    reader = Reader(is_training=True)

    for _ in range(10):

        value = reader.generate(1)

        image = np.squeeze(value['images'])
        label = np.squeeze(value['labels'])

        image = (image+reader.pixel_means).astype(np.int)
        label = reader.decode_label(label)

        plt.imshow(np.vstack([image, label]))
        plt.show()
