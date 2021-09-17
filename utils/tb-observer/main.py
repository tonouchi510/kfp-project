import json
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'pipeline_name', None,
    'Name of pipeline')

flags.DEFINE_string(
    'bucket', 'mitene-ml-research',
    'GCS bucket name')

flags.DEFINE_string(
    'job_id', 'test',
    'ID for job management.')

flags.DEFINE_string(
    'log_dir', None,
    'Directory where tfboard log is stored.')


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    job_dir = f"gs://{FLAGS.bucket}/tmp/{FLAGS.pipeline_name}/{FLAGS.job_id}/{FLAGS.log_dir}"
    metadata = {
        'outputs' : [{
            'type': 'tensorboard',
            'source': f"{job_dir}",
        }]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

if __name__ == '__main__':
    app.run(main)
