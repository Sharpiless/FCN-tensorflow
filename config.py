import numpy as np
import os

LAYERS_SHAPE = [(14, 14),
                (28, 28),
                (56, 56)]

DATA_PATH = '../VOC2012'

EPOCHES = 500

TRAIN_ABLE = True

NEG_WEIGHTS = 0.5

LOSS_WEIGHTS = (1.0, 1.0, 0.5, 0.4)

LEARNING_RATE = 2e-4

ImageSets_PATH = os.path.join(DATA_PATH, 'ImageSets')

TARGET_SIZE = 224

KEEP_RATE = 0.8

BATCH_SIZE = 4

BATCHES = 256

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

LAYERS = (

    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',

    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',

    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',

    'relu5_3', 'conv5_4', 'relu5_4'

)
