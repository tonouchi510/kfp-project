import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import tensorflow as tf
from absl import app, flags


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
    "batch_size", 64,
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


def build_model(input_dim: int):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=[input_dim]),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(FLAGS.learning_rate)
    model.compile(
        loss='mse',
        optimizer=optimizer,
        metrics=['mae', 'mse'])
    return model

def preprocess(dataset: DataFrame):
    def norm(x):
        train_stats = train_dataset.describe()
        train_stats = train_stats.transpose()
        return (x - train_stats['mean']) / train_stats['std']

    dataset = dataset.dropna()
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1)*1.0
    dataset['Europe'] = (origin == 2)*1.0
    dataset['Japan'] = (origin == 3)*1.0

    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    return (normed_train_data, train_labels), (normed_test_data, test_labels)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    job_dir = f"gs://{FLAGS.bucket}/tmp/{FLAGS.pipeline}/{FLAGS.job_id}/training"
    artifacts_dir = f"gs://{FLAGS.bucket}/artifacts/{FLAGS.pipeline}/{FLAGS.job_id}/training"

    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                        na_values = "?", comment='\t',
                        sep=" ", skipinitialspace=True)
    dataset = raw_dataset.copy()

    (train_x, train_y), (test_x, test_y) = preprocess(dataset)
    input_dim = len(train_x.keys())

    model = build_model(input_dim)
    model.summary()

    tfboard_cb = tf.keras.callbacks.TensorBoard(log_dir=f"{job_dir}/logs", histogram_freq=1)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        train_x, train_y,
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        validation_split=0.2,
        verbose=1,
        callbacks=[tfboard_cb, early_stop]
    )

    loss, mae, mse = model.evaluate(test_x, test_y, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    model.save(f"{artifacts_dir}/saved_model", include_optimizer=True)


if __name__ == "__main__":
    app.run(main)
