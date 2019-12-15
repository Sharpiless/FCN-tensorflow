import numpy as np
import random
import config as cfg


def random_crop(image, label_image=None, with_label=True, size=cfg.TARGET_SIZE):

    h, w = image.shape[:2]

    bias_y = h - size
    bias_x = w - size

    if with_label:

        pos_num = np.sum((label_image != 0)+0)

        num = 0

        while(num < pos_num*0.5):

            y = random.randint(0, bias_y) if bias_y > 0 else 0
            x = random.randint(0, bias_x) if bias_x > 0 else 0

            crop_image = label_image[y:y+size, x:x+size, :]
            num = np.sum((crop_image != 0)+0)

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


def random_crop(image, label_image=None, with_label=True, size=cfg.TARGET_SIZE):

    h, w = image.shape[:2]

    bias_y = h - size
    bias_x = w - size

    pos_num = np.sum((label_image != 0)+0)

    num = 0

    while(num < pos_num*0.5):

        y = random.randint(0, bias_y) if bias_y > 0 else 0
        x = random.randint(0, bias_x) if bias_x > 0 else 0

        crop_image = image[y:y+size, x:x+size, :]
        num = np.sum((label_image != 0)+0)

    image = image[y:y+size, x:x+size, :]

    flip = random.randint(0, 1)

    if flip:
        image = np.flip(image, axis=1)
        label_image = np.flip(label_image, axis=1)

    if with_label:
        label_image = label_image[y:y+size, x:x+size, :]
        return image, label_image

    else:
        return image
