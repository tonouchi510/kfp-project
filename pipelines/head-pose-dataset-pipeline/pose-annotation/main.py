import os
import time
import random
import hashlib
from absl import app
from absl import flags
from typing import Dict, List
from logging import getLogger

import tensorflow as tf
from google.cloud import storage
from google.cloud import vision

logger = getLogger(__name__)


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "pipeline", "head-pose-dataset-pipeline",
    "Name of pipeline")

flags.DEFINE_string(
    "bucket_name", "",
    "GCS bucket name")

flags.DEFINE_string(
    "job_id", "test",
    "ID for job management.")

flags.DEFINE_float(
    "valid_ratio", 0.1,
    "ratio of valid_dataset.")

flags.DEFINE_string(
    "chunk_file", None,
    "gcs path of chunk file.")


def skip_judge(bucket_name: str, destination: str) -> bool:
    """処理のスキップ判定のため、処理済みファイルがすでに存在するかチェックする.

    Args:
        bucket_name (str): 保存先GCSバケット名
        destination (str): 処理済みファイルの保存先（prefix）

    Returns:
        [bool]: スキップするかどうか（処理済みファイルがすでに存在すればTrue）
    """
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=destination)
    exist = None
    for b in blobs:
        exist = b
    return True if exist else False


def download_blob(bucket_name, source_blob_name):
    """Downloads a blob from the bucket if not exist."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"

    destination_file_name = os.path.basename(source_blob_name)
    if os.path.exists(destination_file_name):
        print(f"Already exist: {destination_file_name}")
    else:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def crop_face(filename: str, vertices: List, flip: bool=False):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits)
    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=vertices[0][1],
        offset_width=vertices[0][0],
        target_height=vertices[2][1]-vertices[0][1],
        target_width=vertices[2][0]-vertices[0][0]
    )
    if flip:
        image = tf.image.flip_left_right(image)
    image = tf.image.encode_jpeg(
        image, optimize_size=True, chroma_downsampling=False)
    return image


def download_media(bucket, filepath: str) -> str:
    dest_filename = f"tmp/{os.path.basename(filepath)}"
    try:
        blob = bucket.blob(filepath)
        blob.download_to_filename(dest_filename)
    except Exception as e:
        logger.warning(f"Error: {filepath}")
        logger.exception(e)
    return dest_filename


def to_tfrecord(
    image_bytes: bytes,
    yaw: float,
    pitch: float,
    roll: float,
    filename: bytes
):
    def _bytestring_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    features = {
        "image": _bytestring_feature([image_bytes]),
        "yaw": _float_feature([yaw]),
        "pitch": _float_feature([pitch]),
        "roll": _float_feature([roll]),
        "filename": _bytestring_feature([filename])
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def detect_faces_uri(client, uri: str):
    """Detects faces in the file located in Google Cloud Storage or the web."""
    image = vision.Image()
    image.source.image_uri = uri

    response = client.face_detection(image=image)
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    faces = response.face_annotations
    if not faces:
        return [], None, None, None

    face = faces[0]
    vertices = []
    for v in face.fd_bounding_poly.vertices:
        x = v.x if "x" in v else 0.0
        y = v.y if "y" in v else 0.0
        vertices.append((x, y))
    roll = face.roll_angle
    pan = face.pan_angle
    tilt = face.tilt_angle
    return vertices, roll, pan, tilt


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if FLAGS.chunk_file == "{{item}}":
        logger.info("Task file does not exist, so skip it.")
        exit(0)

    artifacts_dir = f"gs://{FLAGS.bucket_name}/artifacts/{FLAGS.pipeline}/{FLAGS.job_id}"

    download_blob(
        bucket_name=FLAGS.bucket_name, source_blob_name=FLAGS.chunk_file)
    chunk_file = os.path.basename(FLAGS.chunk_file)
    with open(chunk_file, "r") as f:
        target_files = [s.strip() for s in f.readlines()]

    f_number: str = hashlib.md5(str(target_files[0:5]).encode()).hexdigest()
    if random.random() >= FLAGS.valid_ratio:
        fn = f"{artifacts_dir}/train-{f_number}.tfrec"
    else:
        fn = f"{artifacts_dir}/valid-{f_number}.tfrec"
    logger.info(f"Start creating {fn} ...")

    if skip_judge(
        FLAGS.bucket_name,
        destination=fn.replace(f"gs://{FLAGS.bucket_name}/", "")
    ):
        logger.info(f"{f_number} already exist. so skip!")
        exit(0)

    vision_client = vision.ImageAnnotatorClient()
    storage_client = storage.Client()
    bucket = storage_client.bucket(FLAGS.bucket_name)
    os.mkdir("tmp")
    results: Dict[str, Dict] = {}
    for filepath in target_files:
        vertices, roll, pan, tilt = detect_faces_uri(
            vision_client, f"gs://{FLAGS.bucket_name}/{filepath}")
        if roll or pan or tilt:
            dest = download_media(bucket, filepath)
            results[dest] = {
                "vertices": vertices,
                "roll": roll,
                "pan": pan,
                "tilt": tilt
            }
            logger.info(f"{dest}: {str(results[dest])}")
        time.sleep(1)

    with tf.io.TFRecordWriter(fn) as out_file:
        for path, res in results.items():
            try:
                image = crop_face(path, res["vertices"])
                example = to_tfrecord(
                    image_bytes=image.numpy(),
                    yaw=-res["pan"],  # yawとpanの角度は逆向き
                    pitch=res["tilt"],
                    roll=res["roll"],
                    filename=path.encode())
                out_file.write(example.SerializeToString())
            except Exception as e:
                logger.warn(e)
                continue
            
            try:
                # データ拡張（水平フリップ）
                flip_image = crop_face(path, res["vertices"], flip=True)
                flip_example = to_tfrecord(
                    image_bytes=flip_image.numpy(),
                    yaw=res["pan"],  # 符号反転
                    pitch=res["tilt"],
                    roll=-res["roll"],  # 符号反転
                    filename=path.encode())
                out_file.write(flip_example.SerializeToString())
            except Exception as e:
                logger.warn(e)
                continue

    logger.info("Done.")


if __name__ == "__main__":
    app.run(main)
