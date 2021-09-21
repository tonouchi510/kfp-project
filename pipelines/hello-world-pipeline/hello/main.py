from os import error
from absl import app
from absl import flags
import logging

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "message", "hello world",
    "Message string for output console.")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    
    if FLAGS.message:
        logging.INFO(FLAGS.message)
    else:
        logging.ERROR("No massage error.")
        exit(-1)

if __name__ == "__main__":
    app.run(main)