from typing import Callable
import tensorflow as tf


def get_loss_func(alpha=1.0) -> Callable:
    """損失関数はここに記述

    Args:
        alpha (float, optional): _description_. Defaults to 1.0.
    """
    def loss_func(y_true, y_pred):
        """例: 意味はないけどalpha掛けて返すloss関数"""
        l = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        loss = l * alpha
        return loss
    return loss_func
