import tensorflow as tf
from tensorflow.python.keras.engine.functional import Functional
from vit_keras import vit
from typing import Any


def build_model(
    *,
    model_type: str,
    num_classes: int,
    image_size: int,
    optimizer: Any,
    loss: Any,
    pretrained: bool = True,
    checkpoint: str = ""
) -> Functional:
    """モデルの構築はここに記述.

    チェックポイント（ある場合）のロードや、コンパイルも行い、トレーニングループ
    を回す準備が整った状態のtf.keras.Modelのオブジェクトを返す.

    Args:
        model_type (str): 使用するモデルを決定するパラメータ
        num_classes (int): クラス数
        image_size (int): 入力画像サイズ
        optimizer (Any): 使用する最適化アルゴリズム
        loss (Any): 使用する損失関数
        pretrained (bool, optional): 事前学習済みの重みを使用するか. Defaults to True.
        checkpoint (str, optional): 復旧するチェックポイントファイル名. Defaults to "".

    Raises:
        ValueError: _description_

    Returns:
        Functional: fitメソッドを呼び出す準備の整ったtf.keras.Modelのオブジェクト
    """
    if model_type == "vit":
        model = vit.vit_b16(
            image_size=image_size,
            pretrained=pretrained,
            include_top=False,
            pretrained_top=False,
            classes=num_classes
        )
    elif model_type == "efficientnet":
        model = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top=False,
            input_shape=(image_size, image_size, 3),
            weights="imagenet" if pretrained else None,
            classes=num_classes,
        )
    elif model_type == "resnet":
        model = tf.keras.applications.resnet_v2.ResNet50V2(
            include_top=False,
            input_shape=(image_size, image_size, 3),
            weights="imagenet" if pretrained else None,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Invalid argument '{model_type}'.")
    
    if checkpoint:
        model.load_weights(checkpoint)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
