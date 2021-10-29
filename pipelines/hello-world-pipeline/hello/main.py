from absl import app
from absl import flags

from utils.logger import get_logger
logger = get_logger(__name__)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "message", "hello world",
    "Message string for output console.")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    
    if FLAGS.message:
        logger.info(FLAGS.message)
    else:
        logger.error("No massage error.")
        exit(-1)

if __name__ == "__main__":
    app.run(main)
