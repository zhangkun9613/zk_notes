import tensorflow as tf
from tensorflow.python.ops import lookup_ops

import collections

from . import model
from . import iterator
from ..utils import data_utils


class TrainModel(
  collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
  pass

class EvalModel(
  collections.namedtuple("EvalModel", ("graph", "model", "iterator", "word_placeholder",
                                       "pos_placeholder", "batch_size_placeholder"))):
  pass

class Vocabs(
  collections.namedtuple("Vocabs", ("word_vocab_size", "pos_vocab_size", "dst_vocab_size",
                                    "word_vocab", "pos_vocab", "dst_vocab", "reverse_dst_vocab"))):
  pass


def create_vocabs(input_dir):
  word_vocab, _ = data_utils.initialize_vocabulary(input_dir + 'word.vocab')
  pos_vocab, _ = data_utils.initialize_vocabulary(input_dir + 'pos.vocab')
  dst_vocab, reverse_dst_vocab = data_utils.initialize_vocabulary(input_dir + 'label.vocab')
  return Vocabs(word_vocab_size=len(word_vocab),
                pos_vocab_size=len(pos_vocab),
                dst_vocab_size=len(dst_vocab),
                word_vocab=word_vocab,
                pos_vocab=pos_vocab,
                dst_vocab=dst_vocab,
                reverse_dst_vocab=reverse_dst_vocab)


def create_train_model(hparams, input_dir):
  graph = tf.Graph()
  with graph.as_default():
    word_vocab_table = lookup_ops.index_table_from_file(input_dir + 'word.vocab', default_value=0)
    pos_vocab_table = lookup_ops.index_table_from_file(input_dir + 'pos.vocab', default_value=0)
    label_vocab_table = lookup_ops.index_table_from_file(input_dir + 'label.vocab', default_value=0)
    dataset = tf.contrib.data.TextLineDataset(input_dir + 'train.src')
    vocabs = create_vocabs(input_dir)
    it = iterator.get_iterator(dataset, word_vocab_table, pos_vocab_table, label_vocab_table, vocabs.dst_vocab_size, hparams.batch_size)
    m = model.Model(hparams, it, tf.contrib.learn.ModeKeys.TRAIN, vocabs, None)
  return TrainModel(
    graph=graph,
    model=m,
    iterator=it)


def create_eval_model(hparams, input_dir):
  graph = tf.Graph()
  with graph.as_default():
    word_vocab_table = lookup_ops.index_table_from_file(input_dir + 'word.vocab', default_value=0)
    pos_vocab_table = lookup_ops.index_table_from_file(input_dir + 'pos.vocab', default_value=0)
    word_place_holder = tf.placeholder(shape=[None], dtype=tf.string)
    pos_place_holder = tf.placeholder(shape=[None], dtype=tf.string)
    batch_size_place_holder = tf.placeholder(shape=[], dtype=tf.int64)
    word_dataset = tf.contrib.data.Dataset.from_tensor_slices(word_place_holder)
    pos_dataset = tf.contrib.data.Dataset.from_tensor_slices(pos_place_holder)
    vocabs = create_vocabs(input_dir)
    it = iterator.get_infer_iterator(word_dataset, pos_dataset, word_vocab_table, pos_vocab_table, batch_size_place_holder, hparams.max_seq_len)
    reverse_dst_table = lookup_ops.index_to_string_table_from_file(input_dir + 'label.vocab', default_value='UNKNOWN')
    m = model.Model(hparams, it, tf.contrib.learn.ModeKeys.EVAL, vocabs, reverse_dst_table)
  return EvalModel(
    graph=graph,
    model=m,
    iterator=it,
    word_placeholder=word_place_holder,
    pos_placeholder=pos_place_holder,
    batch_size_placeholder=batch_size_place_holder)


def create_serving_model(hparams, input_dir):
  vocabs = create_vocabs(input_dir)
  graph = tf.Graph()
  with graph.as_default():
    #TODO the index_table_from_file can not save these table
    word_vocab_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(list(vocabs.word_vocab.keys()),
                                                  list(vocabs.word_vocab.values())),
      0)
    pos_vocab_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(list(vocabs.pos_vocab.keys()),
                                                  list(vocabs.pos_vocab.values())),
      0)
    word_place_holder = tf.placeholder(shape=[], dtype=tf.string)
    pos_place_holder = tf.placeholder(shape=[], dtype=tf.string)
    it = iterator.get_serving_iterator(word_place_holder, pos_place_holder, word_vocab_table, pos_vocab_table, hparams.max_seq_len)
    reverse_dst_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(list(vocabs.dst_vocab.values()),
                                                  list(vocabs.dst_vocab.keys()),
                                                  key_dtype=tf.int64,
                                                  value_dtype=tf.string),
      'UNKNOWN')
    m = model.Model(hparams, it, tf.contrib.learn.ModeKeys.INFER, vocabs, reverse_dst_table)
  return EvalModel(
    graph=graph,
    model=m,
    iterator=it,
    word_placeholder=word_place_holder,
    pos_placeholder=pos_place_holder,
    batch_size_placeholder=None)


def train(hparams, input_dir, output_dir):
  train_model = create_train_model(hparams, input_dir)
  with tf.Session(graph=train_model.graph) as sess:
    summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
    #ckpt = tf.train.latest_checkpoint(output_dir)
    #train_model.model.saver.restore(sess, ckpt)
    sess.run(tf.tables_initializer())
    sess.run(train_model.iterator.initializer)
    sess.run(tf.global_variables_initializer())
    global_step = train_model.model.global_step.eval(session=sess)
    while global_step < hparams.num_train_steps:
      try:
        _, _, _, step_summary, global_step = train_model.model.train(sess)
      except tf.errors.OutOfRangeError:
        sess.run(train_model.iterator.initializer)
        continue
      summary_writer.add_summary(step_summary, global_step)
      if global_step > 0 and global_step % 400 == 0:
        train_model.model.saver.save(sess, output_dir + 'model.ckpt', global_step=global_step)
        eval(hparams, input_dir, output_dir) 
    #train_model.model.saver.save(sess, output_dir + 'model.ckpt', global_step=global_step)
    summary_writer.close()


def eval(hparams, input_dir, output_dir):
  with tf.gfile.GFile(input_dir + 'dev.src', mode='r') as f:
    data = f.read().splitlines()
    data = [d.split("\t") for d in data]
  eval_model = create_eval_model(hparams, input_dir)
  with tf.Session(graph=eval_model.graph) as sess:
    ckpt = tf.train.latest_checkpoint(output_dir)
    eval_model.model.saver.restore(sess, ckpt)
    sess.run(tf.tables_initializer())
    sess.run(eval_model.iterator.initializer, feed_dict={
      eval_model.word_placeholder: [d[0] for d in data],
      eval_model.pos_placeholder: [d[1] for d in data],
      eval_model.batch_size_placeholder: len(data)
    })
    result = eval_model.model.eval(sess)
    result = [r.decode('utf-8') for r in result[0]]
    count = 0
    target = [d[2] for d in data]
    for i in range(len(data)):
      if target[i] != result[i]:
        count = count + 1
        print(data[i], target[i], result[i])
    print((len(data) - count) / len(data))


def export(hparams, input_dir, output_dir, export_dir):
  eval_model = create_serving_model(hparams, input_dir)
  with tf.Session(graph=eval_model.graph) as sess:
    ckpt = tf.train.latest_checkpoint(output_dir)
    eval_model.model.saver.restore(sess, ckpt)
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    word_placeholder_info = tf.saved_model.utils.build_tensor_info(eval_model.word_placeholder)
    pos_placeholder_info = tf.saved_model.utils.build_tensor_info(eval_model.pos_placeholder)
    output_info = tf.saved_model.utils.build_tensor_info(eval_model.model.predictions)
    prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
          'words': word_placeholder_info,
          'pos': pos_placeholder_info,
        },
        outputs={
          'outputs': output_info,
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
      )
    )
    init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
        'domain_classify': prediction_signature
      },
      legacy_init_op=init_op)
    builder.save()
