import os
import json
import logging
from absl import app
from absl import flags
from google.cloud import storage

PIPELINE_NAME = "hello-world-pipeline"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'bucket', 'xxxx',
    'GCS bucket name')

flags.DEFINE_string(
    "job_id", "test",
    "ID for job management.")

flags.DEFINE_string(
    "message_file", None,
    "GCS path to the file where the message was recorded.")


def download_blob(bucket_name, source_blob_name) -> str:
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

        print(
            "Blob {} downloaded to {}.".format(
                source_blob_name, destination_file_name
            )
        )
    return destination_file_name

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    
    logging.info(f"Start: {PIPELINE_NAME} {FLAGS.job_id}")

    source_blob_name = FLAGS.message_file.replace(f"gs://{FLAGS.bucket}/", "")
    filename = download_blob(FLAGS.bucket, source_blob_name)

    outputs = []
    with open(filename, "r") as f:
        outputs.append(f.readline())
    logging.info(f"Num of messages: {len(outputs)}")

    with open("/tmp/out.json", "w") as f:
        json.dump(outputs, f, indent=4)


if __name__ == '__main__':
    app.run(main)
