import numpy as np
import random
import config as cfg


def random_crop(image, label_image, with_label=True, size=cfg.TARGET_SIZE):

    h, w = image.shape[:2]

    bias_y = h - size
    bias_x = w - size

    y = random.randint(0, bias_y) if bias_y > 0 else 0
    x = random.randint(0, bias_x) if bias_x > 0 else 0

    image = image[y:y+size, x:x+size, :]
    
    if with_label:
        label_image = label_image[y:y+size, x:x+size, :]

    return image, label_image
