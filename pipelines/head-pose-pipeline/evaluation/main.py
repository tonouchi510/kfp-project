import csv
import os
import numpy as np
import tensorflow as tf
from typing import List
from absl import app, flags
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
    "pipeline", None,
    "Name of pipeline")

flags.DEFINE_string(
    "bucket_name", None,
    "GCS bucket name")

flags.DEFINE_string(
    "job_id", "test",
    "ID for job management.")

flags.DEFINE_integer(
    "model_type", 5,
    "Model type for evaluation.")

flags.DEFINE_string(
    "test_dataset_name", "BIWI",
    "open dataset name for evaluation.")

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
    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))
    return destination_file_name


def create_npydata_pipeline(
    x: np.ndarray,
    y: np.ndarray,
):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    option = tf.data.Options()

    option.experimental_deterministic = True
    dataset = (
        dataset.with_options(option)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    artifacts_dir = f"gs://{FLAGS.bucket_name}/artifacts/{FLAGS.pipeline}/{FLAGS.job_id}/evaluation"

    weight_file = download_blob(
        bucket_name=FLAGS.bucket_name,
        source_blob_name=f"artifacts/{FLAGS.pipeline}/{FLAGS.job_id}/training/weights.h5"
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

    test_path = download_blob(
        bucket_name=FLAGS.bucket_name,
        source_blob_name=f"datasets/{FLAGS.test_dataset_name}/{FLAGS.test_dataset_name}_test.npz"
    )
    test_data = np.load(test_path)
    x_test, y_test = test_data["image"], test_data["pose"]
    dataset = create_npydata_pipeline(x_test, y_test)

    diffs = []
    for img, gt_pose in dataset:
        img = np.expand_dims(img.numpy(), 0)
        gt_pose = gt_pose.numpy()
        pred = model(img).numpy()[0]
        diffs.append(np.abs(pred - gt_pose))

    mae = round(np.mean(diffs), 3)
    pose_matrix = np.mean(diffs, axis=0)
    pose_matrix = np.round(pose_matrix, decimals=4)
    roll_mae, pitch_mae, yaw_mae = pose_matrix
    print(f"MAE: {mae}")
    print(pose_matrix)

    with open("results.csv", "w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([roll_mae, pitch_mae, yaw_mae, mae])

    storage_client = storage.Client()
    bucket = storage_client.bucket(FLAGS.bucket_name)
    blob = bucket.blob(f"{artifacts_dir.replace(f'gs://{FLAGS.bucket_name}/', '')}/results.csv")
    blob.upload_from_filename("results.csv")

    v = KFPVisualization(FLAGS.pipeline, FLAGS.bucket_name, FLAGS.job_id)
    v.produce_ui_metadata_table(
        source=f"{artifacts_dir}/results.csv", header=["roll", "pitch", "yaw", "mae"])
    v.write_ui_metadata()

    v.produce_metrics(name="mae", value=mae, f="RAW")
    v.write_metrics()


if __name__ == "__main__":
    app.run(main)
