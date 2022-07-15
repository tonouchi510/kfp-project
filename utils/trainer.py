# DLトレーニングで共通のロジック・モジュール

import tensorflow as tf
from tensorflow.python.data.ops.readers import TFRecordDatasetV2
from google.cloud import storage
import pandas as pd
from typing import Callable, List
import os


def get_tfrecord_dataset(
    dataset_path: str,
    preprocessing: Callable,
    global_batch_size: int,
    split: str,
    data_augmentation: Callable = lambda x, y: (x, y),
) -> TFRecordDatasetV2:
    """TFRecordからデータパイプラインを構築する.

    Args:
        dataset_path (str): 目的のTFRecordファイルが保存されているパス.
        preprocessing (Callable): 適用する前処理関数.
        global_batch_size (int): バッチサイズ(分散処理の場合は合計).
        split (str): train or valid
        data_augmentation (Callable, optional): データオーグメンテーション関数. Defaults to lambdax:x.

    Raises:
        FileNotFoundError: dataset_pathにファイルが存在しない場合.

    Returns:
        TFRecordDatasetV2: 定義済みのデータパイプライン.
    """
    # Build a pipeline
    file_names = tf.io.gfile.glob(
        f"{dataset_path}/{split}-*.tfrec"
    )
    dataset = tf.data.TFRecordDataset(
        file_names, num_parallel_reads=tf.data.AUTOTUNE)
    if not file_names:
        raise FileNotFoundError(f"Not found: {dataset}")

    option = tf.data.Options()
    if split == "train":
        option.experimental_deterministic = False
        dataset = dataset.with_options(option) \
            .map(lambda example: preprocessing(example=example), num_parallel_calls=tf.data.AUTOTUNE) \
            .map(lambda x, y: data_augmentation(x, y)) \
            .shuffle(512, reshuffle_each_iteration=True) \
            .batch(global_batch_size, drop_remainder=True) \
            .prefetch(tf.data.AUTOTUNE)
    else:
        option.experimental_deterministic = True
        dataset = dataset.with_options(option) \
            .map(lambda example: preprocessing(example=example), num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(global_batch_size, drop_remainder=False) \
            .prefetch(tf.data.AUTOTUNE)
    return dataset


class CheckpointCallBack(tf.keras.callbacks.Callback):
    """gcsへのアップロードも含めたCheckpointCallBack

    ※savedModelと違い、h5pyファイルは直接gcsに書き込めないので
    """
    def __init__(
        self,
        checkpoint_dir: str,
        storage_client: storage.Client,
        bucket_name: str,
    ):
        super(CheckpointCallBack, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.bucket = storage_client.bucket(bucket_name)

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(f"{epoch:05d}.h5")
        blob = self.bucket.blob(f"{self.checkpoint_dir}/{epoch:05d}.h5")
        blob.upload_from_filename(f"{epoch:05d}.h5")


class Training:
    def __init__(
        self,
        build_model_func: Callable,
        job_dir: str,
        artifacts_dir: str = "",
        use_tpu: bool = True,
    ) -> None:
        """トレーニングの初期設定を行う.

        TPUノードの管理、TPUStrategyの設定、モデルのロード、コンパイル、checkpointの復旧などを行う.

        Arguments:
            build_model_func (Callable): 実験に使うモデルのbuild関数を渡す.
            job_dir (str): job管理用のGCSパス. checkpointやlogの保存をする.
            artifacts_dir (str): 実験結果の保存先GCSパス.
            use_tpu (bool): トレーニングにTPUを使うかどうか.
        """
        # For job management
        self.job_dir = job_dir
        self.artifacts_dir = artifacts_dir
        self.use_tpu = use_tpu
        self.storage_client = storage.Client()
        self.bucket_name = self.job_dir.split("/")[2]
        self.last_epoch = self._get_last_epoch()

        if self.use_tpu:
            # Tpu cluster setup
            cluster = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(cluster)
            tf.tpu.experimental.initialize_tpu_system(cluster)
            self.distribute_strategy = tf.distribute.TPUStrategy(cluster)

            # Load model in distribute_strategy scope
            with self.distribute_strategy.scope():
                self._setup_model(build_model=build_model_func)
        else:
            self._setup_model(build_model=build_model_func)

        self.callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=f"{self.job_dir}/logs", histogram_freq=1),
            tf.keras.callbacks.TerminateOnNaN(),
            CheckpointCallBack(
                checkpoint_dir=os.path.join(self.job_dir.replace(f"gs://{self.bucket_name}/", ""), "checkpoints"),
                storage_client=self.storage_client,
                bucket_name=self.bucket_name,
            )
        ]

    def _setup_model(self, build_model: Callable) -> None:
        if self.last_epoch == 0:
            self.model = build_model()
        else:
            checkpoint = f"{self.last_epoch:0>5}.h5"
            self.model = build_model(checkpoint=checkpoint)

    def _get_last_epoch(self) -> int:
        dest = self.job_dir.replace(f"gs://{self.bucket_name}/", "")
        blobs = self.storage_client.list_blobs(self.bucket_name, prefix=f"{dest}/checkpoints")
        checkpoints = [0]
        for b in blobs:
            epoch = b.name.replace(f"{dest}/checkpoints/", "").split(".")[0]
            if epoch:
                checkpoints.append(int(epoch))
        last_epoch = max(checkpoints)
        if last_epoch > 0:
            # last_epochのチェックポイントファイルをダウンロード
            blob = self.storage_client.bucket(self.bucket_name).blob(f"{dest}/checkpoints/{last_epoch:0>5}.h5")
            blob.download_to_filename(f"{last_epoch:0>5}.h5")
        return last_epoch

    def add_callbacks(self, callbacks: List) -> None:
        self.callbacks.extend(callbacks)

    def run_train(
        self,
        train_ds: TFRecordDatasetV2,
        valid_ds: TFRecordDatasetV2,
        epochs: int
    ):
        """トレーニングを実施し、ログや結果を保存する.

        tf.keras.Model.fitでのトレーニングを行う.
        複雑なトレーニングループが必要な場合もtf.keras.Model.train_stepをオーバーライドするなどして使う.

        Arguments:
            train_ds (TFRecordDatasetV2): tensorflowのデータセットパイプライン（学習用）.
            valid_ds (TFRecordDatasetV2): tensorflowのデータセットパイプライン（検証用）.
            epochs (int): トレーニングを回す合計エポック数.
        """
        self.history = self.model.fit(
            train_ds,
            validation_data=valid_ds,
            callbacks=self.callbacks,
            initial_epoch=self.last_epoch,
            epochs=epochs
        )
        if self.artifacts_dir:
            self.model.save(f"{self.artifacts_dir}/saved_model", include_optimizer=False)
        return

    def save_history(self):
        """tf.keras.Model.fitの返却物であるHistoryをGCSにアップロードする.

        実験結果の確認や、optuna-pipelineでのチューニングに使用される.
        """
        hist_df = pd.DataFrame(self.history.history)
        hist_df.to_csv("history.csv")
        bucket = self.storage_client.bucket(self.bucket_name)
        blob = bucket.blob(f"{self.artifacts_dir.replace(f'gs://{self.bucket_name}/', '')}/history.csv")
        blob.upload_from_filename("history.csv")
