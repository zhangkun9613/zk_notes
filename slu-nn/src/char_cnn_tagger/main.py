import tensorflow as tf
import os
import json
import logging
from src.char_cnn_tagger import train

tf.app.flags.DEFINE_string("hparams_dir", "../../hparams/def_char_cnn_tagger.json", "hyper parameters")
tf.app.flags.DEFINE_string("input_dir", "../../data/char_cnn_tagger/", "train data dir")
tf.app.flags.DEFINE_string("output_dir", "../../output/char_cnn_tagger/", "log/model output dir")

FLAGS = tf.app.flags.FLAGS
hparams = tf.contrib.training.HParams()

def loadAndCheck():
  with tf.gfile.GFile(FLAGS.hparams_dir, "r") as f:
    d = json.loads(f.read())
    for k, v in d.items(): hparams.add_hparam(k, v)
  logging.info("hparams file loads success")
  if not os.path.isdir(FLAGS.input_dir):
    raise IOError("Input dir: %s is not dir" % (FLAGS.input_dir, ))
  if not tf.gfile.Exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

def run_train(unused_argv):
  loadAndCheck()
  train.train(hparams, FLAGS.input_dir, FLAGS.output_dir)

def run_eval(unused_argv):
  loadAndCheck()
  train.eval(hparams, FLAGS.input_dir, FLAGS.output_dir)

if __name__ == "__main__":
  # tf.app.run(main=run_train)
  tf.app.run(main=run_eval)

