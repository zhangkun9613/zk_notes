import tensorflow as tf

import os
import shutil
import sys
import argparse
sys.path.append('../..')

from src.tagger import train

tf.app.flags.DEFINE_string("hparams_file", "../../hparams/def_tagger.json", "overridden hyper parameters")
tf.app.flags.DEFINE_string("input_dir", "../../training/tagging/", "training data dir")
tf.app.flags.DEFINE_string("output_dir", "../../output/tagging", "log/model dir")
tf.app.flags.DEFINE_string("export_dir", "../../export/tagging", "serving model export dir")
tf.app.flags.DEFINE_string("export_version", "1", "serving model version")
tf.app.flags.DEFINE_string("domain", "media", "choose domain")
tf.app.flags.DEFINE_boolean("train", False, "is_train")
tf.app.flags.DEFINE_boolean("test", False, "is_test")
tf.app.flags.DEFINE_boolean("export", False, "is_export")
FLAGS = tf.app.flags.FLAGS

default_command_hparams = tf.contrib.training.HParams(
  num_train_steps=2000,
  learning_rate=0.001,
  word_embedding_size=300,
  pos_embedding_size=100,
  max_seq_len=20,
  dropout_keep_prob=0.3,
  batch_size=32,
  fasttext_model='../../data/fasttext_model/zh.bin',
  fasttext_wv_size=300,
  rnn_size=400,
  attention_option=None # attetion_options: luong, bahdanau, None
)

default_media_hparams = tf.contrib.training.HParams(
  num_train_steps=10000,
  learning_rate=0.0005,
  word_embedding_size=300,
  pos_embedding_size=100,
  max_seq_len=20,
  dropout_keep_prob=0.3,
  batch_size=50,
  fasttext_model='../../data/fasttext_model/zh.bin',
  fasttext_wv_size=300,
  rnn_size=400,
  attention_option=None # attetion_options: luong, bahdanau, None
)

default_question_hparams = tf.contrib.training.HParams(
  num_train_steps=10000,
  learning_rate=0.0005,
  word_embedding_size=300,
  pos_embedding_size=100,
  max_seq_len=20,
  dropout_keep_prob=0.3,
  batch_size=50,
  fasttext_model='../../data/fasttext_model/zh.bin',
  fasttext_wv_size=300,
  rnn_size=400,
  attention_option=None # attetion_options: luong, bahdanau, None
)

default_spec_question_hparams = tf.contrib.training.HParams(
  num_train_steps=3000,
  learning_rate=0.001,
  word_embedding_size=300,
  pos_embedding_size=100,
  max_seq_len=20,
  dropout_keep_prob=0.5,
  batch_size=50,
  # fasttext_model='/work/ml/wordvectors/zh.bin',
  fasttext_model='../../data/fasttext_model/zh.bin',
  fasttext_wv_size=300,
  rnn_size=256,
  attention_option='luong' # attetion_options: luong, bahdanau, None
)


domain = FLAGS.domain
output_dir = FLAGS.output_dir + "_" + domain + "/"
export_dir = FLAGS.export_dir + "_" + domain + "/"

def load_hparams():
  if domain == "media":
    hparams = default_media_hparams
  elif domain == "command":
    hparams = default_command_hparams
  elif domain == "question":
    hparams = default_question_hparams
  else:
    hparams = default_spec_question_hparams
  # with tf.gfile.GFile(FLAGS.hparams_file, "r") as f:
  #  hparams.parse_json(f.read())
  return hparams


def run_train(unused_argv):
  hparams = load_hparams()
  if not os.path.isdir(FLAGS.input_dir):
    raise IOError("no input directory found")  # python2.7 not support FileNotFoundError
    # raise FileNotFoundError("no input directory found")
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  train.train(hparams, FLAGS.input_dir, output_dir, domain)


def run_eval(unused_argv):
  hparams = load_hparams()
  if not os.path.isdir(FLAGS.input_dir):
    raise IOError("no input directory found")# python2.7 not support FileNotFoundError
    # raise FileNotFoundError("no input directory found")
  if not os.path.isdir(output_dir):
    raise IOError("no input directory found")  # python2.7 not support FileNotFoundError
    # raise FileNotFoundError("no output directory found")
  train.eval(hparams, FLAGS.input_dir, output_dir, domain)


def run_export(unused_argv):
  hparams = load_hparams()
  version_export_dir = os.path.join(export_dir, FLAGS.export_version)
  if os.path.exists(version_export_dir):
    shutil.rmtree(version_export_dir)
  train.export(hparams, FLAGS.input_dir, output_dir, version_export_dir, domain)


CUDA_VISIBLE_DEVICES = 0,1
if __name__ == "__main__":
  if FLAGS.train:
    tf.app.run(main=run_train)
  if FLAGS.test:
    tf.app.run(main=run_eval)
  if FLAGS.export:
    tf.app.run(main=run_export)
