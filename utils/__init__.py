"""
Trainingの汎用化したパッケージ.

新しく作成するTrainingコンポーネントの下に、generalとリネームし、コピーされる。
"""

from .trainer import *
from .logger import *

__all__ = (
    "Training",
    "get_logger",
)
