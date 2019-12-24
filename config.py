import numpy as np
import os

LAYERS_SHAPE = [(14, 14),
                (28, 28),
                (56, 56)]

DATA_PATH = '../VOC2012'

EPOCHES = 500

TRAIN_ABLE = True

NEG_WEIGHTS = 1.0

LOSS_WEIGHTS = (0., 0., 1.0, 0.)

LEARNING_RATE = 1e-5

ImageSets_PATH = os.path.join(DATA_PATH, 'ImageSets')

TARGET_SIZE = 224

KEEP_RATE = 0.8

BATCH_SIZE = 4

BATCHES = 128

MIN_CROP_RATIO = 0.6

MAX_CROP_RATIO = 1.0

MIN_CROP_POS_RATIO = 0.5

PIXEL_MEANS = np.array([[[103.939, 116.779, 123.68]]])

MODEL_PATH = './model/'

CLASSES = ['', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',

           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',

           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',

           'train', 'tvmonitor']

COLOR_MAP = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0),

    (128, 128, 0), (0, 0, 128), (128, 0, 128),

    (0, 128, 128), (128, 128, 128), (64, 0, 0),

    (192, 0, 0), (64, 128, 0), (192, 128, 0),

    (64, 0, 128), (192, 0, 128), (64, 128, 128),

    (192, 128, 128), (0, 64, 0), (128, 64, 0),

    (0, 192, 0), (128, 192, 0), (0, 64, 128)
]
