import tensorflow as tf
from tensorflow.python.ops import lookup_ops

import collections

from . import model
from . import iterator
from ..utils import data_utils


class TrainModel(
  collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
  pass

class ServingModel(
  collections.namedtuple("ServingModel", ("graph", "model", "iterator",
                                          "word_placeholder", "pos_placeholder", "wv_placeholder"))):
  pass

class Vocabs(
  collections.namedtuple("Vocabs", ("word_vocab_size", "pos_vocab_size", "dst_vocab_size",
                                    "word_vocab", "pos_vocab", "dst_vocab", "reverse_dst_vocab"))):
  pass


def conlleval(p, g, w, filename):
  out = ''
  for sl, sp, sw in zip(g, p, w):
    out += ' '.join(sw) + '\n' + ' '.join(sl) + ' labels\n' + ' '.join(sp) + ' predictions\n'
    #for wl, wp, w in zip(sl, sp, sw):
    #  out += w + ' ' + wl + ' ' + wp + '\n'
    #out += 'EOS O O\n\n'

  f = open(filename, 'w')
  f.write(out[:-1])  # remove the ending \n on last line
  f.close()


def create_vocabs(input_dir, domain):
  word_vocab, _ = data_utils.initialize_vocabulary(input_dir + 'word_' + domain + '.vocab')
  pos_vocab, _ = data_utils.initialize_vocabulary(input_dir + 'pos_' + domain + '.vocab')
  dst_vocab, reverse_dst_vocab = data_utils.initialize_vocabulary(input_dir + 'tag_' + domain + '.vocab')
  return Vocabs(word_vocab_size=len(word_vocab),
                pos_vocab_size=len(pos_vocab),
                dst_vocab_size=len(dst_vocab),
                word_vocab=word_vocab,
                pos_vocab=pos_vocab,
                dst_vocab=dst_vocab,
                reverse_dst_vocab=reverse_dst_vocab)


def create_train_model(hparams, input_dir, domain):
  graph = tf.Graph()
  with graph.as_default():
    vocabs = create_vocabs(input_dir, domain)
    it = iterator.Iterator(
      word_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      pos_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      wv_input=tf.placeholder(tf.float32, [None, hparams.max_seq_len, hparams.fasttext_wv_size]),
      dst_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len + 2]),
      dst_output=tf.placeholder(tf.int32, [None, hparams.max_seq_len + 2]),
      src_seq_len=tf.placeholder(tf.int32, [None]),
      dst_seq_len=tf.placeholder(tf.int32, [None]),
      batch_size=hparams.batch_size)
    m = model.Model(hparams, vocabs, it, tf.contrib.learn.ModeKeys.TRAIN, None)
  return TrainModel(
    graph=graph,
    model=m,
    iterator=it)


def create_eval_model(hparams, input_dir, domain):
  graph = tf.Graph()
  with graph.as_default():
    vocabs = create_vocabs(input_dir, domain)
    it = iterator.Iterator(
      word_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      pos_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      wv_input=tf.placeholder(tf.float32, [None, hparams.max_seq_len, hparams.fasttext_wv_size]),
      dst_input=None,
      dst_output=None,
      src_seq_len=tf.placeholder(tf.int32, [None]),
      dst_seq_len=None,
      batch_size=1)
    reverse_dst_vocab = lookup_ops.index_to_string_table_from_file(
      input_dir + 'tag_' + domain + '.vocab', default_value='<unk>')
    m = model.Model(hparams, vocabs, it, tf.contrib.learn.ModeKeys.EVAL, reverse_dst_vocab)
  return TrainModel(
    graph=graph,
    model=m,
    iterator=it)


def create_serving_model(hparams, input_dir, domain):
  graph = tf.Graph()
  with graph.as_default():
    vocabs = create_vocabs(input_dir, domain)
    word_vocab_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(list(vocabs.word_vocab.keys()),
                                                  list(vocabs.word_vocab.values())),
      0)
    pos_vocab_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(list(vocabs.pos_vocab.keys()),
                                                  list(vocabs.pos_vocab.values())),
      0)
    reverse_dst_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(list(vocabs.dst_vocab.values()),
                                                  list(vocabs.dst_vocab.keys()),
                                                  key_dtype=tf.int64,
                                                  value_dtype=tf.string),
      'O')
    word_place_holder = tf.placeholder(shape=[], dtype=tf.string)
    wv_place_holder = tf.placeholder(shape=[None, hparams.fasttext_wv_size], dtype=tf.float32)
    pos_place_holder = tf.placeholder(shape=[], dtype=tf.string)
    it = iterator.get_serving_iterator(word_place_holder, wv_place_holder, pos_place_holder,
                                       word_vocab_table, pos_vocab_table, hparams.max_seq_len)
    m = model.Model(hparams, vocabs, it, tf.contrib.learn.ModeKeys.INFER, reverse_dst_table)
    return ServingModel(
      graph=graph,
      model=m,
      iterator=it,
      word_placeholder=word_place_holder,
      pos_placeholder=pos_place_holder,
      wv_placeholder=wv_place_holder)


def train(hparams, input_dir, output_dir, domain):
  data, _, _, _ = iterator.load_data(hparams.fasttext_model, input_dir, domain, "train", hparams.max_seq_len)
  train_model = create_train_model(hparams, input_dir, domain)
  with tf.Session(graph=train_model.graph) as sess:
    summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    #global_step = train_model.model.global_step.eval(session=sess)
    batch_num = int(hparams.num_train_steps/(len(data)/hparams.batch_size))
    print(batch_num)
    batches = iterator.batch_iterator(data, hparams.batch_size, batch_num)
    for cur_step, batch in enumerate(batches):
      (_, loss, step_summary, global_step) = train_model.model.train(sess, batch)
      # print (train_model.model.logging)
      print('step:'+str(cur_step), 'loss:'+str(loss))
      summary_writer.add_summary(step_summary, global_step)
      if cur_step > 0 and cur_step % 500 == 0:
        train_model.model.saver.save(sess, output_dir + 'model.ckpt', global_step=global_step)
    summary_writer.close()


def eval(hparams, input_dir, output_dir, domain):
  prefix = "train"
  data, _, _, _ = iterator.load_data(hparams.fasttext_model, input_dir, domain, prefix, hparams.max_seq_len)
  eval_model = create_eval_model(hparams, input_dir, domain)
  with tf.Session(graph=eval_model.graph) as sess:
    ckpt = tf.train.latest_checkpoint(output_dir)
    eval_model.model.saver.restore(sess, ckpt)
    sess.run(tf.tables_initializer())
    batches = iterator.batch_iterator(data, len(data), 1, shuffle=False)
    for batch in batches:
      predictions = eval_model.model.eval(sess, batch)[0]
  tags = []
  for i in range(len(predictions)):
    predict = predictions[i].tolist()
    try:
      idx = predict.index(b'</s>')
      predict = predict[:idx]
    except ValueError:
      pass
    tags.append([l.decode('utf-8') for l in predict])
  with tf.gfile.GFile(input_dir + prefix + "_" + domain + ".src", mode='r') as f:
    data = [d.split("\t") for d in f.read().splitlines()]
    words = [d[0]+ ' O' for d in data]
    # words = [line + " intent" for line in words]
    targets = [d[2] + ' O' for d in data]
    for i in range(len(tags)):
      tag = targets[i].split()[0:-1]
      if tag != tags[i]:
        print(data[i],tags[i])
  conlleval(
    tags,
    [t.split() for t in targets],
    [w.split() for w in words],
    output_dir + "tagging.result"
  )


def export(hparams, input_dir, output_dir, export_dir, domain):
  serve_model = create_serving_model(hparams, input_dir, domain)
  with tf.Session(graph=serve_model.graph) as sess:
    ckpt = tf.train.latest_checkpoint(output_dir)
    serve_model.model.saver.restore(sess, ckpt)
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    word_placeholder_info = tf.saved_model.utils.build_tensor_info(serve_model.word_placeholder)
    pos_placeholder_info = tf.saved_model.utils.build_tensor_info(serve_model.pos_placeholder)
    wv_placeholder_info = tf.saved_model.utils.build_tensor_info(serve_model.wv_placeholder)
    output_info = tf.saved_model.utils.build_tensor_info(serve_model.model.predictions)
    prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
          'words': word_placeholder_info,
          'pos': pos_placeholder_info,
          'wv': wv_placeholder_info,
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
        'tagging': prediction_signature
      },
      legacy_init_op=init_op)
    builder.save()
