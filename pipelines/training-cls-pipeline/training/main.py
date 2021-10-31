import functools

from typing import List
import numpy as np
import tensorflow as tf
from absl import app, flags
from utils.trainer import Training, get_labels, get_tfrecord_dataset
from utils.data import preprocess_image, TFRECORD_FEATURES
from utils.logger import get_logger
from tensorflow.python.keras.engine.functional import Functional

logger = get_logger(__name__)

# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "pipeline", None,
    "Name of pipeline")

flags.DEFINE_string(
    "bucket", None,
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
    "learning_rate", 0.0001,
    "Initial learning rate per batch size of 256.")

flags.DEFINE_string(
    "dataset", None,
    "Directory where dataset is stored.")

flags.DEFINE_string(
    "model_type", "efficientnet",
    "Model type for training.")

flags.DEFINE_integer(
    "input_size", 224,
    "size of image.")


def build_model(*, model_type: str,
                input_size: int,
                num_classes: int) -> Functional:

    if model_type == "resnet":
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(input_size, input_size, 3)
        )
    elif model_type == "efficientnet":
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(input_size, input_size, 3)
        )
    else:
        raise ValueError(f"Invalid argument '{model_type}'.")
    head = tf.keras.layers.Dense(num_classes, name="head", activation="softmax")(base_model.output)
    model = tf.keras.Model(inputs=base_model.input, outputs=head)
    return model


def read_tfrecord(example,
                  size: int,
                  label_list: List):
    # decode the TFRecord
    example = tf.io.parse_single_example(example, TFRECORD_FEATURES)
    image = tf.image.decode_png(example["image"], channels=3)
    image = preprocess_image(image, size)
    label = tf.cast(tf.where(label_list == example["label"])[0], tf.float32)

    return image, label


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    job_dir = f"gs://{FLAGS.bucket}/tmp/{FLAGS.pipeline}/{FLAGS.job_id}/training"
    artifacts_dir = f"gs://{FLAGS.bucket}/artifacts/{FLAGS.pipeline}/{FLAGS.job_id}/training"

    # この辺もパイプラインパラメータ化するのもアリ
    #num_classes, label_list = get_labels(FLAGS.dataset)
    num_classes = 100
    label_list = []
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    loss = "sparse_categorical_crossentropy"

    build_model_func = functools.partial(
        build_model,
        model_type=FLAGS.model_type,
        num_classes=num_classes,
        input_size=FLAGS.input_size
    )
    t = Training(
        build_model_func=build_model_func,
        job_dir=job_dir,
        artifacts_dir=artifacts_dir,
        use_tpu=True,
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )
    t.model_summary()

    read_tfrecord_func = functools.partial(
        read_tfrecord,
        size=FLAGS.input_size,
        label_list=label_list
    )

    train_ds = get_tfrecord_dataset(
        dataset_path=FLAGS.dataset,
        preprocessing=read_tfrecord_func,
        global_batch_size=FLAGS.global_batch_size,
        is_train=True
    )
    valid_ds = get_tfrecord_dataset(
        dataset_path=FLAGS.dataset,
        preprocessing=read_tfrecord_func,
        global_batch_size=FLAGS.global_batch_size,
        is_train=False
    )

    t.run_train(train_ds, valid_ds, FLAGS.epochs)
    logger.info(f"End of training. model path is {FLAGS.artifacts_dir}/saved_model")


if __name__ == "__main__":
    app.run(main)
