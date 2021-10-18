"""
Trainingの汎用化したパッケージ.

新しく作成するTrainingコンポーネントの下に、generalとリネームし、コピーされる。
"""

from .logger import *

__all__ = (
    "get_logger",
)
