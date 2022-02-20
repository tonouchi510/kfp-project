import json
import hashlib
from absl import app
from absl import flags
from google.cloud import storage

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "pipeline", None,
    "Name of pipeline")

flags.DEFINE_string(
    "job_id", "test",
    "ID for job management.")

flags.DEFINE_string(
    "bucket_name", "",
    "GCS bucket name")

flags.DEFINE_string(
    "dataset", None,
    "Directory where dataset is stored.")

N_FILES_BY_TFRECORD = 1000


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    job_dir = f"tmp/{FLAGS.pipeline}/{FLAGS.job_id}/data-chunk-spliter"

    client = storage.Client()
    bucket = client.bucket(FLAGS.bucket_name)

    prefix = FLAGS.dataset.replace(f"gs://{FLAGS.bucket_name}/", "")
    blobs = bucket.list_blobs(prefix=prefix)
    files = []
    for b in blobs:
        files.append(b.name)
    n_files = len(files)

    i = 0
    chunk_files = []
    while i < n_files:
        targets = []
        j = 0
        while (i + j) < n_files and j < N_FILES_BY_TFRECORD:
            targets.append(files[i + j])
            j += 1
        f_number: str = hashlib.md5(str(targets[0:5]).encode()).hexdigest()
        chunk_file = f"chunk_{f_number}.txt"
        with open(chunk_file, "w") as f:
            for id in targets:
                f.write("%s\n" % id)
        blob = bucket.blob(f"{job_dir}/{chunk_file}")
        blob.upload_from_filename(chunk_file)
        chunk_files.append(f"{job_dir}/{chunk_file}")
        i += j

    with open("/tmp/out.json", "w") as f:
        json.dump(chunk_files, f, indent=4)


if __name__ == '__main__':
    app.run(main)
