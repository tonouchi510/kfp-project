import functools

from typing import List
import numpy as np
import tensorflow as tf
from absl import app, flags
from utils import (BCTModel, Training, get_labels,
                     get_train_ds, get_valid_ds, loss_func)
from tensorflow.python.keras.engine.functional import Functional


# Random seed fixation
tf.random.set_seed(666)
np.random.seed(666)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'job_id', 'test',
    'ID for job management.')

flags.DEFINE_integer(
    'global_batch_size', 2048,
    'Batch size for training/eval before distribution.')

flags.DEFINE_integer(
    'epochs', 30,
    'Number of epochs to train for.')

flags.DEFINE_float(
    'learning_rate', 0.0001,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_string(
    'dataset', None,
    'Directory where dataset is stored.')

flags.DEFINE_string(
    'model_type', 'vit',
    'Model type for training.')

flags.DEFINE_integer(
    'emb_dim', 512,
    'output dim of embedding layer.')


def build_model(*, num_classes,
                bct_model_path: str,
                bct_exclude_datasets: List,
                emb_dim: int = 512) -> Functional:
    if FLAGS.model_type == "vit":
        model = get_vit(
            num_classes=num_classes,
            image_size=112,
            bct_model_path=bct_model_path,
            bct_exclude_datasets=bct_exclude_datasets,
            emb_dim=emb_dim)
    elif FLAGS.model_type == "arcface":
        model = get_arcface_resnet(num_classes=num_classes,
                                   bct_model_path=bct_model_path,
                                   emb_dim=emb_dim)
    else:
        raise ValueError(f"Invalid argument '{FLAGS.model_type}'.")
    return model


def read_tfrecord(example,
                  size: int,
                  label_list: List,
                  num_classes: int):
    # decode the TFRecord
    example = tf.io.parse_single_example(example, TFRECORD_FEATURES)
    image = tf.image.decode_png(example["image"], channels=3)
    image = preprocess_image(image, size)
    label = tf.cast(tf.where(label_list == example['label'])[0], tf.float32)
    dataset_id = example["dataset_id"]

    if FLAGS.model_type == "vit":
        return image, label, dataset_id
    elif FLAGS.model_type == "arcface":
        label_one_hot = tf.one_hot(tf.cast(label[0], tf.int32), num_classes)
        return (image, label_one_hot), label


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    job_dir = f"gs://{FLAGS.bucket}/tmp/{FLAGS.pipeline}/{FLAGS.job_id}/training"
    artifacts_dir = f"gs://{FLAGS.bucket}/artifacts/{FLAGS.pipeline}/{FLAGS.job_id}/training"

    num_classes, label_list = get_labels(FLAGS.dataset, 100)
    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    loss = loss_func if FLAGS.bct_model_path else "sparse_categorical_crossentropy"
    bct_model_path = FLAGS.bct_model_path
    custom_objects = get_custom_objects(FLAGS.model_type)
    bct_exclude_datasets = [int(s) for s in FLAGS.bct_exclude_datasets.split(",")]

    build_model_func = functools.partial(
        build_model,
        num_classes=num_classes,
        bct_model_path=bct_model_path,
        bct_exclude_datasets=bct_exclude_datasets,
        emb_dim=FLAGS.emb_dim
    )
    t = Training(
        build_model_func=build_model_func,
        job_dir=FLAGS.job_dir,
        artifacts_dir=FLAGS.artifacts_dir,
        use_tpu=True,
        custom_model_class=BCTModel if bct_model_path else None,
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
        custom_objects=custom_objects
    )
    t.model_summary()

    if FLAGS.model_type == "arcface":
        image_size = t.model.input_shape[0][1]
    else:
        image_size = t.get_model_input_size()

    read_tfrecord_func = functools.partial(read_tfrecord,
                                           size=image_size,
                                           label_list=label_list,
                                           num_classes=num_classes)

    train_ds = get_train_ds(
        FLAGS.dataset, read_tfrecord_func, 32, 100)
    valid_ds = get_valid_ds(
        FLAGS.dataset, read_tfrecord_func, 32, 100)

    t.run_train(train_ds, valid_ds, FLAGS.epochs)

    with open("/tmp/out.txt", "w") as f:
        f.write(f"{FLAGS.artifacts_dir}/saved_model")


if __name__ == '__main__':
    app.run(main)
