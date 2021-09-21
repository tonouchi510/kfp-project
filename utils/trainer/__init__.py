"""
Trainingの汎用化したパッケージ.

新しく作成するTrainingコンポーネントの下に、generalとリネームし、コピーされる。
"""

from .trainer import *
from .bct import *

__all__ = (
    "Training",
    "get_train_ds",
    "get_valid_ds",
    "get_labels",
    "BCTModel",
    "DistillBCTModel",
    "loss_func"
)
