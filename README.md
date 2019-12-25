
## 主要分几个程序：
#### 1.config.py：保存了我们这个项目的大部分参数；
#### 2.read_data.py：读取VOC数据集中用于的语义分割的训练集及其标签图片；
#### 3.random_crop.py：随机裁剪和缩放图片；
#### 4.fcn_vgg.py：定义了FCN的网络结构并训练模型；
#### 5.fcn_vgg_restore.py：用于加载预训练模型；


### 1.config.py
config.py保存了大部分参数，先上代码：

```python
import numpy as np
import os

# 反卷积层的大小
LAYERS_SHAPE = [(14, 14),
                (28, 28),
                (56, 56)]

#数据集路径
DATA_PATH = '../VOC2012'

#迭代次数
EPOCHES = 500

# 前馈网络是否参与训练
TRAIN_ABLE = True

# 背景分类的loss权重
NEG_WEIGHTS = 0.5

# 不同反卷积层的loss权重
LOSS_WEIGHTS = (1.0, 1.0, 0., 0.)

# 学习率
LEARNING_RATE = 2e-5

ImageSets_PATH = os.path.join(DATA_PATH, 'ImageSets')

TARGET_SIZE = 224

KEEP_RATE = 0.8

BATCH_SIZE = 4

BATCHES = 256

# 最小缩放比率
MIN_CROP_RATIO = 0.6

# 最大缩放比例
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

```
这里没有什么要多说的，只是要注意一点：我们会使用4个反卷积层，最后一个反卷积层即是原图大小；其中这四个反卷积层都要参与loss的计算，因此设置了不同的权重；

这里先训练前两个反卷积层，因此后两个反卷积层的权重先置为零；

### 2.read_data.py
这里定义了一个Reader类，与SSD（上一期）的Reader类相似；

先上代码：

```python
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
        # label = reader.decode_label(label)

        plt.imshow(label)
        plt.show()

```
这是我们读取到的图片（经过裁剪和标准化）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191223200609558.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)
以及每个反卷积层对应的标签图片：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191223200640630.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20191223200652521.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191223200710766.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191223200726727.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)
大小分别为14×14，28×28，56×56，224×224；
### 3.random_crop.py
这里用于对原图和标签图进行随即缩放和裁剪，使用的是最邻近插值法；

代码：

```python
import numpy as np
import random
import config as cfg


def random_crop(image, label_image=None, with_label=True, size=cfg.TARGET_SIZE):

    min_ratio=cfg.MIN_CROP_POS_RATIO

    h, w = image.shape[:2]

    bias_y = h - size
    bias_x = w - size

    if with_label:

        pos_num = np.sum((label_image != 0)+0)

        num = 0

        i = 1

        while(num < pos_num*min_ratio):

            y = random.randint(0, bias_y) if bias_y > 0 else 0
            x = random.randint(0, bias_x) if bias_x > 0 else 0

            crop_image = label_image[y:y+size, x:x+size, :]
            num = np.sum((crop_image != 0)+0)

            i += 1

            if i%5 == 0:
                min_ratio /= 1.2

        image = image[y:y+size, x:x+size, :]
        label_image = label_image[y:y+size, x:x+size, :]

        flip = random.randint(0, 1)

        if flip:
            image = np.flip(image, axis=1)
            label_image = np.flip(label_image, axis=1)

        return image, label_image

    else:

        y = random.randint(0, bias_y) if bias_y > 0 else 0
        x = random.randint(0, bias_x) if bias_x > 0 else 0
        image = image[y:y+size, x:x+size, :]

        return image

```

### 4.fcn_vgg.py
这里定义了一个Net类，实现了FCN网络和训练并保存模型；

代码：

```python
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

        self.x = tf.placeholder(
            tf.float32, [None, self.target_size, self.target_size, 3])

        self.y0 = tf.placeholder(
            tf.int32, [None, 14, 14])
        self.y1 = tf.placeholder(
            tf.int32, [None, 28, 28])
        self.y2 = tf.placeholder(
            tf.int32, [None, 56, 56])
        self.y3 = tf.placeholder(
            tf.int32, [None, self.target_size, self.target_size])

        self.y = [self.y0, self.y1, self.y2, self.y3]

        self.y_hat = self.network(self.x)

        self.loss = self.sample_loss(self.y, self.y_hat)

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

            loss = tf.losses.compute_weighted_loss(loss, pos_mask)
            losses.append(loss)

        # losses = tf.reduce_mean(losses)
        losses = tf.losses.compute_weighted_loss(
            losses, self.loss_weights
        )

        return losses

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
                                 self.y0: labels[0],
                                 self.y1: labels[1],
                                 self.y2: labels[2],
                                 self.y3: labels[3]}

                    _, loss, pred = sess.run(
                        [self.train_step, self.loss, self.y_hat], feed_dict)

                    loss_list.append(loss)

                    # print('batch:{} loss:{}'.format(batch, loss), end='\r')

                loss_values = np.array(loss_list)  # (64, 3)

                loss_values = np.mean(loss_values)

                with open('./result.txt', 'a') as f:
                    f.write(str(loss_values)+'\n')

                self.saver.save(sess, os.path.join(
                    cfg.MODEL_PATH, 'model.ckpt'))

                print('epoch:{} loss:{}'.format(
                    self.reader.epoch, loss_values))


if __name__ == "__main__":

    net = Net(is_training=True)

    for _ in range(4):                
    	net.run_test()
    	
    net.train_net()

```
需要注意的是我们这里使用了ssd_vgg的预训练模型，命名空间也是按照SSD来的；

然后就是tensorflow并没有给出双线性插值初始化的方法，我们就自己定义了get_deconv_filter方法，用于获取双线性差值初始化的卷积核用于反卷积层；

### 5.fcn_cgg_restore.py
用于加载ssd_vgg的预训练模型。

单独写一个是因为我们只需要第一次加载的时候，载入ssd的部分参数，所以训练前需要首先运行一次这个程序，当然如果不加载预训练模型训练的话也可以直接运行fcn_vgg.py，但可能导致模型的loss不收敛；

代码：

```python
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

```
这样就可以开心地训练啦~

（Ps：吐槽一下这个可能要训练3~5天，加上我的电脑配置不高1050Ti+晚上宿舍断电，，，很难受呀，，，先放一下训练一个白天的效果吧，完全训练完之后再来更新：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191223203305639.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191223203855857.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191223204020807.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)
## 我们下期再见~

欢迎关注留言哦~
