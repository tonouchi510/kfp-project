# DLトレーニングで共通のロジック・モジュール

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.data.ops.readers import TFRecordDatasetV2
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.callbacks import History
from typing import Callable, Any
from typing import List, Dict, Tuple
import json


def get_tfrecord_dataset(
    dataset_path: str,
    preprocessing: Callable,
    global_batch_size: int,
    is_train: bool
) -> TFRecordDatasetV2:
    # Build a pipeline
    option = tf.data.Options()
    option.experimental_deterministic = False
    
    if is_train:
        file_names = tf.io.gfile.glob(f"{dataset_path}/train-*.tfrec")
        dataset = tf.data.TFRecordDataset(
            file_names, num_parallel_reads=tf.data.AUTOTUNE)
        dataset = (
            dataset
                .with_options(option)
                .map(lambda example: preprocessing(example=example),
                    num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(512, reshuffle_each_iteration=True)
                .batch(global_batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE)
        )
    else:
        file_names = tf.io.gfile.glob(f"{dataset_path}/valid-*.tfrec")
        dataset = tf.data.TFRecordDataset(
            file_names, num_parallel_reads=tf.data.AUTOTUNE)
        dataset = (
            dataset
                .with_options(option)
                .map(lambda example: preprocessing(example=example),
                    num_parallel_calls=tf.data.AUTOTUNE)
                .batch(global_batch_size, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE)
        )
    return dataset


class Training:
    def __init__(
        self, *,
        build_model_func: Callable,
        job_dir: str = "",
        artifacts_dir: str = "",
        use_tpu: bool = True
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
        self.last_epoch, self.last_checkpoint = self._get_metadata()

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

        tboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=f"{self.job_dir}/logs", histogram_freq=1)
        self.callbacks = [tboard_callback]

    def _setup_model(self, build_model: Callable) -> None:
        if self.last_epoch == 0:
            self.model = build_model()
        else:
            checkpoint = f"{self.job_dir}/checkpoints/{self.last_checkpoint}"
            self.model = tf.keras.models.load_model(checkpoint)

    def _get_metadata(self) -> Tuple[int, str]:
        if file_io.file_exists_v2(f"{self.job_dir}/job_meta.json"):
            with file_io.FileIO(f"{self.job_dir}/job_meta.json", "r") as reader:
                job_meta = json.load(reader)
            return job_meta["last_epoch"], job_meta["last_checkpoint"]
        else:
            meta = {
                "last_epoch": 0,
                "last_checkpoint": "",
            }
            with file_io.FileIO(f"{self.job_dir}/job_meta.json", "w") as writer:
                json.dump(meta, writer)
            return 0, ""

    def _save_metadata(self) -> None:
        with file_io.FileIO(f"{self.job_dir}/job_meta.json", "r") as reader:
            job_meta = json.load(reader)
            job_meta["last_epoch"] = self.last_epoch
            job_meta["last_checkpoint"] = self.last_checkpoint
        with file_io.FileIO(f"{self.job_dir}/job_meta.json", "w+") as writer:
            json.dump(job_meta, writer)

    def model_summary(self) -> None:
        self.model.summary()

    def get_model_input_size(self) -> int:
        return self.model.input_shape[1]

    def add_callback(self, new_callback) -> None:
        self.callbacks.append(new_callback)

    def run_train(
        self,
        train_ds: TFRecordDatasetV2,
        valid_ds: TFRecordDatasetV2,
        epochs: int
    ) -> History:
        """トレーニングを実施し、ログや結果を保存する.

        tf.keras.Model.fitでのトレーニングを行う.
        複雑なトレーニングループが必要な場合もtf.keras.Model.train_stepをオーバーライドするなどして使う.

        Arguments:
            train_ds (TFRecordDatasetV2): tensorflowのデータセットパイプライン（学習用）.
            valid_ds (TFRecordDatasetV2): tensorflowのデータセットパイプライン（検証用）.
            epochs (int): トレーニングを回す合計エポック数.
        """
        for epoch in range(self.last_epoch, epochs):
            history = self.model.fit(train_ds, validation_data=valid_ds,
                                     callbacks=self.callbacks,
                                     initial_epoch=epoch,
                                     epochs=epoch+1)

            self.last_epoch += 1
            self.last_checkpoint = f"{self.last_epoch:0>5}"
            self.model.save(
                f"{self.job_dir}/checkpoints/{self.last_checkpoint}", include_optimizer=True)
            self._save_metadata()

        if self.artifacts_dir:
            self.model.save(f"{self.artifacts_dir}/saved_model", include_optimizer=False)

        return history
