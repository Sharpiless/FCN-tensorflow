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

def normalize(image):

    image = image.astype(np.float)

    min_value = 0
    max_value = 255

    return image/255
