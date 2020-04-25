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

        self.layers = cfg.LAYERS_SHAPE

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

    def standardize(self, image):

        mean = np.mean(image)
        var = np.mean(np.square(image-mean))

        image = (image - mean)/np.sqrt(var)

        return image

    def normalize(self, image):

        return image - cfg.PIXEL_MEANS

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

    def pre_process(self):

        true_labels = self.load_labels()

        if self.is_training:
            np.random.shuffle(true_labels)

        self.true_labels = true_labels

    def get_layers_label(self, label_image):

        results = []

        for layer in self.layers:
            tmp = cv2.resize(label_image, layer,
                             interpolation=cv2.INTER_NEAREST)
            results.append(tmp)

        results.append(label_image)

        return results

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
            label = self.get_layers_label(label)
            raw = image
            image = self.standardize(image)

            images.append(image)
            labels.append(label)

        images = np.stack(images)
        # labels = np.stack(labels)
        result_labels = []
        for i in range(4):
            tmp = []
            for j in range(batch_size):
                tmp.append(labels[j][i])
            result_labels.append(np.stack(tmp))

        value = {'images': images, 'labels': result_labels}

        return value, raw.astype(np.int)

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

        label_image = label_image.astype(np.int)
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


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    reader = Reader(is_training=True)

    for _ in range(10):

        value, raw = reader.generate(1)

        image = np.squeeze(value['images'])
        label = value['labels']
        label_image = np.squeeze(label[-1]).astype(np.float)

        result = reader.decode_label(label_image, np.zeros(shape=raw.shape))

        plt.imshow(result)
        plt.show()
