# general/trainer

トレーニングコンポーネントのベースイメージに使うモジュール。  
汎用的に使うものを共通化して用意してある。

## 内容
- 実験管理
  - `gs://{GCP_PROJECT_ID}/tmp/{パイプライン名}/{JOB_ID}`以下で管理
  - TPUのプリエンプトなどの異常終了時の再試行処理の管理
    - checkpointファイルの保存・復旧、last_epochの記録など
  - トレーニングログの保存
    - 別途tb-observerコンポーネントから読み取り、tensorboardで可視化
  - 学習済みモデル
    - `gs://{GCP_PROJECT_ID}/artifacts/{パイプライン名}/{JOB_ID}`にsaved_model形式で保存
- データパイプラインの構築
  - trainデータとvalidデータのパイプライン構築用のベースを用意してある

## export
- Training:
  - トレーニングコンポーネントの汎用クラス  
  - 初期化時に実験に必要な各種設定、パラメータを渡す
  - カスタムトレーニングループが必要な場合はサブクラスを作成して使用する
    - 必要に応じて`run_train`や`__init__`を修正
- get_tfrecord_dataset
  - TFRecordのパス、入力サイズ、前処理用の関数のpartialを渡す


## サンプルコード（コンポーネント側）
```python
import tensorflow as tf
from general import Training, get_train_ds, get_valid_ds

def build_model(???) -> (str, int):
    """
    各手法固有のコードになるため、コンポーネント側で実装する。
    トレーニングに使うmodelを構築する。必要に応じてカスタムモデルクラスでラップする。
    """
    model = tf.keras.applications.ResNet50(num_classes=num_classes, weights=None, input_shape=(112, 112, 3))
    return model


def read_tfrecord(example, input_size):
    """
    tensorflow dataset パイプラインに組み込む前処理用関数。
    ここも手法に依存するため、個別実装。以下例。
    """
    example = tf.io.parse_single_example(example, TFRECORD_FEATURES)
    image = tf.image.decode_image(example['image'], channels=3)
    image = preprocess_image(image, size)
    label_num = tf.cast(tf.where(label_list == example['label'])[0], tf.float32)
    return image, label_num


def main():
    num_classes, label_list = get_labels(FLAGS.dataset, FLAGS.percentage)

    build_model_func = functools.partial(build_model, num_classes=num_classes)
    t = Training(build_model_func=build_model_func, job_dir=FLAGS.job_dir, artifacts_dir=FLAGS.artifacts_dir,
                 use_tpu=True, custom_model_class=None, optimizer=optimizer, loss=loss, metrics=["accuracy"])
    t.model_summary()

    read_tfrecord_func = functools.partial(
        read_tfrecord, size=t.get_model_input_size(), label_list=label_list)
    
    train_ds = get_train_ds(FLAGS.dataset, read_tfrecord_func, FLAGS.global_batch_size, FLAGS.percentage)
    valid_ds = get_valid_ds(FLAGS.dataset, read_tfrecord_func, FLAGS.global_batch_size, FLAGS.percentage)

    t.run_train(train_ds, valid_ds, epochs)
```

## その他
- BCTを行うためのモジュールも追加(bct.py)
  - https://arxiv.org/abs/2003.11942