import functools
import numpy as np
import tensorflow as tf
from absl import app, flags
from logging import getLogger

from utils.trainer import Training, get_tfrecord_dataset
from model import build_model
from data import preprocess_func, data_augmentation_func
from loss import get_loss_func


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

flags.DEFINE_int(
    "num_classes", 100,
    "num classes to classify images into.")

flags.DEFINE_string(
    "dataset", None,
    "Directory where dataset is stored.")

flags.DEFINE_string(
    "model_type", "resnet",
    "Model type for training.")

flags.DEFINE_integer(
    "image_size", 64,
    "size of input image.")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    job_dir = f"gs://{FLAGS.bucket_name}/tmp/{FLAGS.pipeline_name}/{FLAGS.job_id}/training"
    artifacts_dir = f"gs://{FLAGS.bucket_name}/artifacts/{FLAGS.pipeline_name}/{FLAGS.job_id}/training"

    build_model_func = functools.partial(
        build_model,
        model_type=FLAGS.model_type,
        num_classes=FLAGS.num_classes,
        image_size=FLAGS.image_size,
        optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
        loss=get_loss_func(alpha=2.0)
    )
    t = Training(
        build_model_func=build_model_func,
        job_dir=job_dir,
        artifacts_dir=artifacts_dir,
        use_tpu=True,
    )

    read_tfrecord_func = functools.partial(
        preprocess_func,
        size=FLAGS.image_size,
    )

    train_ds = get_tfrecord_dataset(
        dataset_path=FLAGS.dataset,
        preprocessing=read_tfrecord_func,
        global_batch_size=FLAGS.global_batch_size,
        data_augmentation=data_augmentation_func,
        split="train",
    )
    valid_ds = get_tfrecord_dataset(
        dataset_path=FLAGS.dataset,
        preprocessing=read_tfrecord_func,
        global_batch_size=FLAGS.global_batch_size,
        split="valid",
    )

    t.run_train(train_ds, valid_ds, FLAGS.epochs)
    t.save_history()
    logger.info(f"End of training. model path is {artifacts_dir}/saved_model")


if __name__ == "__main__":
    app.run(main)
