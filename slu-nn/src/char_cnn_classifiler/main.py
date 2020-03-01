import tensorflow as tf
import os
import json
import logging
import shutil
import sys
import argparse
sys.path.append('../..')
from src.char_cnn_classifiler import train

tf.app.flags.DEFINE_string("hparams_file", "../../hparams/def_char_cnn_classifier.json", "hyper parameters")
tf.app.flags.DEFINE_string("input_dir", "../../data/char_cnn_classification/", "training data dir")
tf.app.flags.DEFINE_string("output_dir", "../../output/char_cnn_classification/", "log/model dir")
tf.app.flags.DEFINE_string("export_dir", "../../export/char_cnn_classification/", "serving model export dir")
tf.app.flags.DEFINE_string("export_version", "1", "serving model version")
tf.app.flags.DEFINE_boolean("train", False, "is_train")
tf.app.flags.DEFINE_boolean("test", False, "is_test")
tf.app.flags.DEFINE_boolean("export", False, "is_export")

FLAGS = tf.app.flags.FLAGS
hparams = tf.contrib.training.HParams()
hparams.batchsize = 16
def loadAndCheck():
  with tf.gfile.GFile(FLAGS.hparams_file, "r") as f:
    d = json.loads(f.read())
    for k, v in d.items():
      hparams.add_hparam(k, v)
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

def run_export(unused_argv):
  loadAndCheck()
  version_export_dir = os.path.join(FLAGS.export_dir, FLAGS.export_version)
  if os.path.exists(version_export_dir):
    shutil.rmtree(version_export_dir)
  train.export(hparams, FLAGS.input_dir, FLAGS.output_dir, version_export_dir)

if __name__ == "__main__":
  if FLAGS.train:
    tf.app.run(main=run_train)
  if FLAGS.test:
    tf.app.run(main=run_eval)
  if FLAGS.export:
    tf.app.run(main=run_export)
