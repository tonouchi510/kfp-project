import tensorflow as tf
import numpy as np


def preprocess_func(
    example,
    size: int,
    num_classes: int
):
    """前処理はここに記述

    Args:
        example (_type_): _description_
        size (int): _description_

    Returns:
        _type_: _description_
    """
    # decode the TFRecord
    features = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "label": tf.io.FixedLenFeature([], tf.int32),
    }

    example = tf.io.parse_single_example(example, features)
    image = tf.image.decode_png(example["image"], channels=3)
    image /= 255
    image = tf.image.resize_with_pad(x, size, size, method="bilinear", antialias=False)
    label = example["label"]
    label = tf.one_hot(label, num_classes)
    return image, label


def data_augmentation_func(x, y):
    """データオーグメンテーションはここに記述

    Args:
        x (_type_): image
        y (_type_): label
    
    ※引数は学習データによって異なるので目的のものに書き換える
    """
    def augment_image(image):
        rand_r = np.random.random()
        h, w, c = image.get_shape()
        dn = np.random.randint(15, size=1)[0]+1
        if  rand_r < 0.25:
            image = tf.image.random_crop(image, size=[h-dn, w-dn, c])
            image = tf.image.resize(image, size=[h, w])
        elif rand_r >= 0.25 and rand_r < 0.75:
            image = tf.image.resize_with_crop_or_pad(image, h+dn, w+dn)
            image = tf.image.random_crop(image, size=[h, w, c])
        return image

    return augment_image(x), y

