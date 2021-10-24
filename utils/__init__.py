"""
Trainingの汎用化したパッケージ.

新しく作成するTrainingコンポーネントの下に、generalとリネームし、コピーされる。
"""

from .logger import *
from .data import *
from .trainer import *

__all__ = (
    "get_logger",
    "preprocess_image",
    "TFRECORD_FEATURES",
    "get_labels",
    "get_tfrecord_dataset",
    "Training"
)
