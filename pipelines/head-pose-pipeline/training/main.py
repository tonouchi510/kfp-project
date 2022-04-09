import os
import functools
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import List
from absl import app, flags
from logging import getLogger
from google.cloud import storage

from utils.trainer import Training, get_tfrecord_dataset
from util import DecayLearningRate
import models


logger = getLogger(__name__)

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "pipeline_name", None,
    "Name of pipeline")

flags.DEFINE_string(
    "bucket_name", None,
    "GCS bucket name")

flags.DEFINE_string(
    "job_id", "test",
    "ID for job management.")

flags.DEFINE_integer(
    "global_batch_size", 1024,
    "Batch size for training/eval before distribution.")

flags.DEFINE_integer(
    "epochs", 30,
    "Number of epochs to train for.")

flags.DEFINE_float(
    "learning_rate", 0.001,
    "Initial learning rate per batch size of 256.")

flags.DEFINE_string(
    "dataset", None,
    "Directory where dataset is stored.")

flags.DEFINE_integer(
    "model_type", 2,
    "Model type for training.")

flags.DEFINE_integer(
    "image_size", 64,
    "size of input image.")


def build_model(
    model_type: str,
    num_classes: int,
    optimizer,
    image_size: int,
    stage_num: List[int],
    lambda_d: int,
    checkpoint: str = "",
):
    if model_type == 0:
        model = models.SSR_net_ori_MT(image_size, num_classes, stage_num, lambda_d)()

    elif model_type == 1:
        model = models.SSR_net_MT(image_size, num_classes, stage_num, lambda_d)()

    elif model_type == 2:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7 * 3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model = models.FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    elif model_type == 3:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7 * 3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model = models.FSA_net_Var_Capsule(
            image_size, num_classes, stage_num, lambda_d, S_set
        )()

    elif model_type == 4:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8 * 8 * 3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model = models.FSA_net_noS_Capsule(
            image_size, num_classes, stage_num, lambda_d, S_set
        )()

    elif model_type == 5:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7 * 3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model = models.FSA_net_NetVLAD(image_size, num_classes, stage_num, lambda_d, S_set)()

    elif model_type == 6:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7 * 3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model = models.FSA_net_Var_NetVLAD(
            image_size, num_classes, stage_num, lambda_d, S_set
        )()

    elif model_type == 7:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8 * 8 * 3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model = models.FSA_net_noS_NetVLAD(
            image_size, num_classes, stage_num, lambda_d, S_set
        )()
    
    elif model_type == 8:
        # ベースラインのResNet
        inputs = tf.keras.layers.Input((image_size, image_size, 3))
        base_model = tf.keras.applications.resnet50.ResNet50(
            include_top=False, weights="imagenet", input_tensor=None,
            )
        h = base_model(inputs, training=True)
        h = tf.keras.layers.GlobalAveragePooling2D()(h)
        head = tf.keras.layers.Dense(num_classes, name="head", activation="softmax")(h)
        model = tf.keras.models.Model(inputs=inputs, outputs=head)

    else:
        raise ValueError("Invalid model_type")

    if checkpoint:
        model.load_weights(checkpoint)
    model.compile(optimizer=optimizer, loss=["mae"], loss_weights=[1])
    model.summary()

    return model


def read_tfrecord(
    example,
    size: int,
):
    # decode the TFRecord
    features = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "roll": tf.io.FixedLenFeature([], tf.float32),
        "pitch": tf.io.FixedLenFeature([], tf.float32),
        "yaw": tf.io.FixedLenFeature([], tf.float32)
    }

    example = tf.io.parse_single_example(example, features)
    x = tf.image.decode_png(example["image"], channels=3)
    x /= 255
    x = tf.image.resize_with_pad(x, size, size, method="bilinear", antialias=False)
    yaw = example["yaw"]
    pitch = example["pitch"]
    roll = example["roll"]
    return x, [yaw, pitch, roll]


def download_blob(bucket_name, source_blob_name):
    """Downloads a blob from the bucket if not exist."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    destination_file_name = os.path.basename(source_blob_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))
    return destination_file_name


def data_augment_pose(x,y):
    def augment_data(image):
        rand_r = np.random.random()
        h, w, c = image.get_shape()
        dn = np.random.randint(15,size=1)[0]+1
        if  rand_r < 0.25:
            image = tf.image.random_crop(image, size=[h-dn, w-dn, c])
            image = tf.image.resize(image, size=[h, w])
        elif rand_r >= 0.25 and rand_r < 0.75:
            image = tf.image.resize_with_crop_or_pad(image, h+dn, w+dn)
            image = tf.image.random_crop(image, size=[h, w, c])
        """
        if np.random.random() > 0.3:
            image = tf.keras.preprocessing.image.random_zoom(image, [0.8,1.2], row_axis=0, col_axis=1, channel_axis=2)
        """
        return image

    return augment_data(x), y


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    job_dir = f"gs://{FLAGS.bucket_name}/tmp/{FLAGS.pipeline_name}/{FLAGS.job_id}/training"
    artifacts_dir = f"gs://{FLAGS.bucket_name}/artifacts/{FLAGS.pipeline_name}/{FLAGS.job_id}/training"

    start_decay_epoch = [30, 60]
    stage_num = [3, 3, 3]
    lambda_d = 1
    num_classes = 3
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)

    build_model_func = functools.partial(
        build_model,
        model_type=FLAGS.model_type,
        num_classes=num_classes,
        stage_num=stage_num,
        lambda_d=lambda_d,
        image_size=FLAGS.image_size,
        optimizer=optimizer,
    )
    t = Training(
        build_model_func=build_model_func,
        job_dir=job_dir,
        artifacts_dir=artifacts_dir,
        use_tpu=True,
    )
    t.add_callbacks([DecayLearningRate(start_decay_epoch)])

    read_tfrecord_func = functools.partial(
        read_tfrecord,
        size=FLAGS.image_size,
    )

    train_ds = get_tfrecord_dataset(
        dataset_path=FLAGS.dataset,
        preprocessing=read_tfrecord_func,
        global_batch_size=FLAGS.global_batch_size,
        data_augmentation=data_augment_pose,
        split="train",
    )
    valid_ds = get_tfrecord_dataset(
        dataset_path=FLAGS.dataset,
        preprocessing=read_tfrecord_func,
        global_batch_size=FLAGS.global_batch_size,
        split="valid",
    )

    history = t.run_train(train_ds, valid_ds, FLAGS.epochs)
    if history:
        hist_df = pd.DataFrame(history.history)
        hist_df.to_csv("history.csv")

        storage_client = storage.Client()
        bucket = storage_client.bucket(FLAGS.bucket_name)
        blob = bucket.blob(f"{artifacts_dir.replace(f'gs://{FLAGS.bucket_name}/', '')}/history.csv")
        blob.upload_from_filename("history.csv")
    
        t.model.save_weights(f"weights.h5")
        blob = bucket.blob(f"{artifacts_dir.replace(f'gs://{FLAGS.bucket_name}/', '')}/weights.h5")
        blob.upload_from_filename("weights.h5")

    logger.info(f"End of training. model path is {artifacts_dir}/saved_model")


if __name__ == "__main__":
    app.run(main)
