# みてねでのDLトレーニングで共通のロジック・モジュール

import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.data.ops.readers import TFRecordDatasetV2
from tensorflow.python.keras.engine.functional import Functional
from tensorflow.python.keras.callbacks import History
from typing import Callable, Any
from typing import List, Dict, Tuple
import json
from google.cloud import storage

BUCKET_NAME = "mitene-ml-research"


def get_labels(dataset: str, percentage: int) -> Tuple[int, List]:
    """データセットの教師ラベルを取得する.

    引数datasetとpercentageを組み合わせて適切なデータセットを選択する.
    percentageが100出ない時は対応するsubsetデータセットから取得する.

    Args:
        dataset (str): データセットのgcsパス
        percentage (int): データセットの何%サブセットを使用するか

    Returns:
        [int]: データセットのクラス数
        [List]: ラベルのリスト
    """

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    prefix = dataset.replace(f"gs://{BUCKET_NAME}/", "")
    if percentage == 100:
        source_blob_name = f"{prefix}/labels.txt"
    else:
        source_blob_name = f"{prefix}-subset-{percentage}/labels.txt"
    dest_filename = "./labels.txt"

    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(dest_filename)

    label_list = []
    with open(dest_filename) as f:
        label_list = [s.strip() for s in f.readlines()]
    num_classes = len(label_list)

    return num_classes, label_list


def get_tfrecord_dataset(dataset_path: str,
                 preprocessing: Callable,
                 global_batch_size: int,
                 is_train: bool) -> TFRecordDatasetV2:
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
    def __init__(self, *,
                 build_model_func: Callable,
                 job_dir: str = "",
                 artifacts_dir: str = "",
                 use_tpu: bool = True,
                 custom_model_class: Any = None,
                 custom_objects: Dict = None,
                 optimizer: Any = None,
                 loss: Any = None,
                 metrics: Any = None) -> None:
        """トレーニングの初期設定を行う.

        TPUノードの管理、TPUStrategyの設定、モデルのロード、コンパイル、checkpointの復旧などを行う.

        Arguments:
            build_model_func (Callable): 実験に使うモデルのbuild関数を渡す.
            job_dir (str): job管理用のGCSパス. checkpointやlogの保存をする.
            artifacts_dir (str): 実験結果の保存先GCSパス.
            use_tpu (bool): トレーニングにTPUを使うかどうか.
            custom_model_class (any): tf.keras.Modelのサブクラス.使用しない場合はNone.
            optimizer (any): tf.keras.Model.compileに渡すoptimizer.
            loss (any): tf.keras.Model.compileに渡すloss.
            metrics (any): tf.keras.Model.compileに渡すmetricsのリスト.
        """
        # For job management
        self.job_dir = job_dir
        self.artifacts_dir = artifacts_dir
        self.use_tpu = use_tpu
        self.last_epoch, self.last_checkpoint = self._get_metadata()

        self.custom_objects = dict()
        self.custom_objects["loss_func"] = loss
        if custom_objects:
            self.custom_objects.update(custom_objects)

        if self.use_tpu:
            # Tpu cluster setup
            cluster = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(cluster)
            tf.tpu.experimental.initialize_tpu_system(cluster)
            self.distribute_strategy = tf.distribute.TPUStrategy(cluster)

            # Load model in distribute_strategy scope
            with self.distribute_strategy.scope():
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                self._setup_model(build_model=build_model_func,
                                  custom_model_class=custom_model_class,
                                  optimizer=optimizer,
                                  loss=loss,
                                  metrics=metrics)
        else:
            self._setup_model(build_model=build_model_func,
                              custom_model_class=custom_model_class,
                              optimizer=optimizer,
                              loss=loss,
                              metrics=metrics)

        tboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=f"{self.job_dir}/logs", histogram_freq=1)
        self.callbacks = [tboard_callback]

    def _setup_model(self, build_model: Callable, custom_model_class,
                     optimizer, loss, metrics) -> None:
        if self.last_epoch == 0:
            self.model = build_model()
            self.model.compile(optimizer=optimizer,
                               loss=loss,
                               metrics=metrics)
        else:
            checkpoint = f"{self.job_dir}/checkpoints/{self.last_checkpoint}"
            self.model = tf.keras.models.load_model(
                checkpoint, custom_objects=self.custom_objects)
            optimizer = self.model.optimizer

            if custom_model_class:
                self.model = custom_model_class(self.model.inputs, self.model.outputs)
            self.model.compile(optimizer=optimizer,
                               loss=loss,
                               metrics=metrics)

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

    def run_train(self, train_ds: TFRecordDatasetV2,
                  valid_ds: TFRecordDatasetV2, epochs: int) -> History:
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
