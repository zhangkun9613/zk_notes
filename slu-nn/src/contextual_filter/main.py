import tensorflow as tf
import os
import shutil
import logging
import json

from src.contextual_filter import train

tf.app.flags.DEFINE_string("output_dir", "../../output/contextual/", "log/model dir")
tf.app.flags.DEFINE_string("input_dir", "../../data/contextual_filter/", "train/dev/test data")
tf.app.flags.DEFINE_string("export_dir", "../../export/contextual/", "model export dir")
tf.app.flags.DEFINE_string("export_version", "1", "serving model version")
tf.app.flags.DEFINE_string("hparams_dir", "../../hparams/def_contextual_filter.json", "hparams settings")

FLAGS = tf.app.flags.FLAGS
hparams = tf.contrib.training.HParams()

def loadAndCheck():
  with tf.gfile.GFile(FLAGS.hparams_dir, "r") as f:
    d = json.loads(f.read())
    for k, v in d.items(): hparams.add_hparam(k, v)
  logging.info("hparam file loads success")
  if not os.path.isdir(FLAGS.input_dir):
    raise IOError("Input dir: %s is not input dir" % (FLAGS.input_dir,))
  if not tf.gfile.Exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

def run_train(unused_argv):
  loadAndCheck()
  train.train(hparams, FLAGS.input_dir, FLAGS.output_dir)

def run_eval(unused_argv):
  loadAndCheck()
  train.eval(hparams, FLAGS.input_dir, FLAGS.output_dir, 'dev')

def run_export(unused_argv):
  loadAndCheck()
  export_dir = os.path.join(FLAGS.export_dir, FLAGS.export_version)
  if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
  train.export(hparams, FLAGS.input_dir, FLAGS.output_dir, export_dir)

if __name__ == "__main__":
  #tf.app.run(main=run_train)
  #tf.app.run(main=run_eval)
  tf.app.run(main=run_export)