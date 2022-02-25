import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor


# 共通のTFRecordの構造（例）
TFRECORD_FEATURES = {
	"version": tf.io.FixedLenFeature([], tf.float32, default_value=1.0),
    "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "label": tf.io.FixedLenFeature([], tf.string, default_value="")
}

# 共通の前処理関数（例）
def preprocess_image(img: EagerTensor, size: int) -> EagerTensor:
    """tfrecord pipeline中の前処理で使用する画像用の関数

    Args:
        img (EagerTensor): tf.decode_image関数により読み込んだ画像データ
        size (int): リサイズ処理のターゲットサイズ

    Return:
        image (EagerTensor): 前処理後の画像.

    """
    img = tf.image.convert_image_dtype(img, tf.bfloat16)
    img = tf.image.resize_with_pad(
        img, size, size,
        method="bilinear",
        antialias=False
    )
    img = tf.image.per_image_standardization(img)
    return img

