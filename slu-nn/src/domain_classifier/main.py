import tensorflow as tf

import os
import shutil

from src.domain_classifier import train

tf.app.flags.DEFINE_string("hparams_file", "../../hparams/def_domain_classifier.json", "overridden hyper parameters")
tf.app.flags.DEFINE_string("input_dir", "../../training/domain/", "training data dir")
tf.app.flags.DEFINE_string("output_dir", "../../output/domain/", "log/model dir")
tf.app.flags.DEFINE_string("export_dir", "../../export/domain/", "serving model export dir")
tf.app.flags.DEFINE_string("export_version", "1", "serving model version")

FLAGS = tf.app.flags.FLAGS

def load_hparams():
  hparams = tf.contrib.training.HParams(
    num_train_steps=1000000,
    class_weights=[1.0, 1.0, 1.0],
    learning_rate=0.0003,
    word_embedding_size=300,
    pos_embedding_size=100,
    max_seq_len=20,
    max_pool_size=4,
    filter_sizes=[1, 2, 3, 4, 5],
    num_filters=128,
    rnn_size=300,
    train_dropout_keep_prob=0.6,
    batch_size=50
  )
  with tf.gfile.GFile(FLAGS.hparams_file, "r") as f:
    hparams.parse_json(f.read())
  return hparams


def run_train(unused_argv):
  hparams = load_hparams()
  if not os.path.isdir(FLAGS.input_dir):
    raise IOError("no input directory found")  # python2.7 not support FileNotFoundError
    # raise FileNotFoundError("no input directory found")
  if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)
  train.train(hparams, FLAGS.input_dir, FLAGS.output_dir)


def run_eval(unused_argv):
  hparams = load_hparams()
  if not os.path.isdir(FLAGS.input_dir):
    raise IOError("no input directory found")  # python2.7 not support FileNotFoundError
    # raise FileNotFoundError("no input directory found")
  if not os.path.isdir(FLAGS.output_dir):
    raise IOError("no input directory found")  # python2.7 not support FileNotFoundError
    # raise FileNotFoundError("no output directory found")
  train.eval(hparams, FLAGS.input_dir, FLAGS.output_dir)


def run_export(unused_argv):
  hparams = load_hparams()
  export_dir = os.path.join(FLAGS.export_dir, FLAGS.export_version)
  if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
  train.export(hparams, FLAGS.input_dir, FLAGS.output_dir, export_dir)


if __name__ == "__main__":
  tf.app.run(main=run_train)
  #tf.app.run(main=run_eval)
  #tf.app.run(main=run_export)

