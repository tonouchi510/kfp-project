import csv
import os
import cv2
import numpy as np
import tensorflow as tf
from typing import List
from absl import app, flags
from math import cos, sin
from logging import getLogger
from google.cloud import storage

import models
from utils.pipeline import KFPVisualization

logger = getLogger(__name__)

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "pipeline_name", None,
    "Name of pipeline")

flags.DEFINE_string(
    "bucket_name", "",
    "GCS bucket name")

flags.DEFINE_string(
    "job_id", "test",
    "ID for job management.")

flags.DEFINE_integer(
    "model_type", 5,
    "Model type for evaluation.")

flags.DEFINE_string(
    "test_dataset", "",
    "dataset gcs path for evaluation.")

flags.DEFINE_integer(
    "image_size", 64,
    "size of input image.")


def build_model(
    model_type: str,
    num_classes: int,
    image_size: int,
    stage_num: List[int],
    lambda_d: int,
):
    if model_type == 0:
        model = models.SSR_net_ori_MT(image_size, num_classes, stage_num, lambda_d)()

    elif model_type == 1:
        model = models.SSR_net_MT(image_size, num_classes, stage_num, lambda_d)()

    elif model_type == 2:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model = models.FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    elif model_type == 3:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model = models.FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    elif model_type == 4:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8*8*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model = models.FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    elif model_type == 5:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model = models.FSA_net_NetVLAD(image_size, num_classes, stage_num, lambda_d, S_set)()

    elif model_type == 6:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 7*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model = models.FSA_net_Var_NetVLAD(image_size, num_classes, stage_num, lambda_d, S_set)()

    elif model_type == 7:
        num_capsule = 3
        dim_capsule = 16
        routings = 2

        num_primcaps = 8*8*3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model = models.FSA_net_noS_NetVLAD(image_size, num_classes, stage_num, lambda_d, S_set)()
    
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

    return model


def download_blob(bucket_name, source_blob_name):
    """Downloads a blob from the bucket if not exist."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    destination_file_name = os.path.basename(source_blob_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))
    return destination_file_name


def get_dataset(
    dataset_path: str,
    image_size: int
):
    def read_tfrecord(
        example,
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
        x = tf.image.resize_with_pad(x, image_size, image_size, method="bilinear", antialias=False)
        yaw = example["yaw"]
        pitch = example["pitch"]
        roll = example["roll"]
        return x, [yaw, pitch, roll]

    file_names = tf.io.gfile.glob(f"{dataset_path}/valid-*.tfrec")

    dataset = tf.data.TFRecordDataset(file_names, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = (
        dataset.map(lambda x: read_tfrecord(x), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def draw_axis(img: np.ndarray, yaw, pitch, roll, tdx=None, tdy=None, size = 80):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    artifacts_dir = f"gs://{FLAGS.bucket_name}/artifacts/{FLAGS.pipeline_name}/{FLAGS.job_id}/evaluation"
    storage_client = storage.Client()
    bucket = storage_client.bucket(FLAGS.bucket_name)

    weight_file = download_blob(
        bucket_name=FLAGS.bucket_name,
        source_blob_name=f"artifacts/{FLAGS.pipeline_name}/{FLAGS.job_id}/training/weights.h5"
    )

    stage_num = [3, 3, 3]
    lambda_d = 1
    model = build_model(
        model_type=FLAGS.model_type,
        num_classes=3,
        image_size=FLAGS.image_size,
        stage_num=stage_num,
        lambda_d=lambda_d,
    )
    model.load_weights(weight_file)
    model.summary()

    n = 1
    diffs = []
    dataset = get_dataset(dataset_path=FLAGS.test_dataset, image_size=FLAGS.image_size)
    for img, gt_pose in dataset:
        img = np.expand_dims(img.numpy(), 0)
        gt_pose = gt_pose.numpy()
        # prediction
        pred = model(img).numpy()[0]
        diffs.append(np.abs(pred - gt_pose))
        if n <= 50:
            # 50件だけ推論結果を描画してアップロード
            draw_img = draw_axis(img[0] * 255, pred[0], pred[1], pred[2])
            cv2.imwrite(f"{n:05d}.png", draw_img)
            blob = bucket.blob(f"{artifacts_dir.replace(f'gs://{FLAGS.bucket_name}/', '')}/imgs/{n:05d}.png")
            blob.upload_from_filename(f"{n:05d}.png")
            # ground truth
            draw_gt_img = draw_axis(img[0] * 255, gt_pose[0], gt_pose[1], gt_pose[2])
            cv2.imwrite(f"{n:05d}_gt.png", draw_gt_img)
            blob = bucket.blob(f"{artifacts_dir.replace(f'gs://{FLAGS.bucket_name}/', '')}/imgs/{n:05d}_gt.png")
            blob.upload_from_filename(f"{n:05d}_gt.png")
        n += 1
        if n % 1000 == 0:
            logger.info(f"Predict {n} records...")
    logger.info("Done.")

    mae = round(np.mean(diffs), 3)
    pose_matrix = np.mean(diffs, axis=0)
    pose_matrix = np.round(pose_matrix, decimals=4)
    yaw_mae, pitch_mae, roll_mae = pose_matrix
    logger.info(f"MAE: {mae}")
    logger.info(pose_matrix)

    with open("results.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([yaw_mae, pitch_mae, roll_mae, mae])

    blob = bucket.blob(f"{artifacts_dir.replace(f'gs://{FLAGS.bucket_name}/', '')}/results.csv")
    blob.upload_from_filename("results.csv")

    v = KFPVisualization(FLAGS.pipeline_name, FLAGS.bucket_name, FLAGS.job_id)
    v.produce_ui_metadata_table(
        source=f"{artifacts_dir}/results.csv", header=["yaw", "pitch", "roll", "mae"])
    v.write_ui_metadata()

    v.produce_metrics(name="mae", value=float(mae), f="RAW")
    v.write_metrics()


if __name__ == "__main__":
    app.run(main)
