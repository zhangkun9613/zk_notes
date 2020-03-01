import tensorflow as tf
import numpy as np
import collections
from src.char_cnn_tagger import iterator
from src.char_cnn_tagger import model

word_vocab_path = "vocabs/word.vocab"
char_vocab_path = "vocabs/char.vocab"
pos_vocab_path = "vocabs/pos.vocab"
target_vocab_path = "vocabs/target.vocab"

class TrainModel(
  collections.namedtuple("TrainModel", ("model", "graph", "iterator"))):
  pass


class Vocabs(
  collections.namedtuple("Vocabs", ("word_vocab_size", "word_vocab", "rev_word_vocab", "pos_vocab", "pos_vocab_size",
                                    "target_vocab", "target_vocab_size", "rev_target_vocab",
                                    "char_vocab", "rev_char_vocab", "char_vocab_size"))):
  pass

def conlleval(p, g, w, filename):
  out = ''
  for sl, sp, sw in zip(g, p, w):
    out += 'BOS O O\n'
    for wl, wp, w in zip(sl, sp, sw):
      out += w + ' ' + wl + ' ' + wp + '\n'
    out += 'EOS O O\n\n'

  f = open(filename, 'w')
  f.writelines(out[:-1])  # remove the ending \n on last line
  f.close()

def create_vocabs(input_dir):
  word_vocab, rev_word_vocab = \
    iterator.initialize_vocabulary(input_dir + word_vocab_path)
  char_vocab, rev_char_vocab = \
    iterator.initialize_vocabulary(input_dir + char_vocab_path)
  target_vocab, rev_target_vocab = \
    iterator.initialize_vocabulary(input_dir + target_vocab_path)
  pos_vocab, _ = \
    iterator.initialize_vocabulary(input_dir + pos_vocab_path)
  return Vocabs(
    word_vocab=word_vocab,
    rev_word_vocab=rev_word_vocab,
    word_vocab_size=len(word_vocab),
    char_vocab=char_vocab,
    rev_char_vocab=rev_char_vocab,
    char_vocab_size=len(char_vocab),
    target_vocab=target_vocab,
    rev_target_vocab=rev_target_vocab,
    target_vocab_size=len(target_vocab),
    pos_vocab=pos_vocab,
    pos_vocab_size=len(pos_vocab)
  )

def create_train_model(hparams, input_dir):
  graph = tf.Graph()
  with graph.as_default():
    vocabs = create_vocabs(input_dir)
    it = iterator.Iterator(
      word_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      pos_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      char_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len, hparams.max_char_len]),
      target_in=tf.placeholder(tf.int32, [None, hparams.max_seq_len + 2]),
      target_out=tf.placeholder(tf.int32, [None, hparams.max_seq_len + 2]),
      word_len=tf.placeholder(tf.int32, [None]),
      char_len=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      target_len=tf.placeholder(tf.int32, [None]),
      batch_size=hparams.batch_size
    )
    m = model.Model(hparams, vocabs, it, tf.contrib.learn.ModeKeys.TRAIN, None)
    return TrainModel(
      model=m,
      graph=graph,
      iterator=it,
    )

def create_eval_model(hparams, input_dir):
  graph = tf.Graph()
  with graph.as_default():
    vocabs = create_vocabs(input_dir)
    it = iterator.Iterator(
      word_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      pos_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      char_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len, hparams.max_char_len]),
      target_in=None,
      target_out=None,
      word_len=tf.placeholder(tf.int32, [None]),
      char_len=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      target_len=None,
      batch_size=1,
    )
    reverse_target_table = tf.contrib.lookup.index_to_string_table_from_file(input_dir + target_vocab_path)
    m = model.Model(hparams, vocabs, it, tf.contrib.learn.ModeKeys.EVAL, reverse_target_table)
    return TrainModel(
      model=m,
      graph=graph,
      iterator=it
    )

def train(hparams, input_dir, output_dir):
  data = iterator.load_data(
    hparams.fasttext_model, input_dir, hparams.max_char_len, hparams.max_seq_len, "train")
  train_model = create_train_model(hparams, input_dir)
  with tf.Session(graph=train_model.graph) as sess:
    summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    global_step = train_model.model.global_step.eval(session=sess)
    batch_num = int(hparams.num_train_steps / (len(data) / hparams.batch_size))
    batches = iterator.get_batch(data, hparams.batch_size, batch_num)
    for cur_step, batch in enumerate(batches):
      # with open("output/result.txt", "w") as f:
      #   f.write(train_model.model.embedding)
      (_, loss, step_summary, global_step) = train_model.model.train(sess, batch)
      if cur_step % 200 == 0:
        print(cur_step)
        print(loss)
      summary_writer.add_summary(step_summary, global_step)
    train_model.model.saver.save(sess, output_dir + 'model.ckpt', global_step=global_step)
    summary_writer.close()

def eval(hparams, input_dir, output_dir):
  data = iterator.load_data(
    hparams.fasttext_model, input_dir, hparams.max_char_len, hparams.max_seq_len, "dev")
  eval_model = create_eval_model(hparams, input_dir)
  with tf.Session(graph=eval_model.graph) as sess:
    ckpt = tf.train.latest_checkpoint(output_dir)
    eval_model.model.saver.restore(sess, ckpt)
    sess.run(tf.tables_initializer())
    batches = iterator.get_batch(data, len(data), 1, shuffle=False)
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
    with tf.gfile.GFile(input_dir + "dev.in", mode='r') as f:
      data = f.read().splitlines()
      words = data[::3]
      targets = data[2::3]
    conlleval(
      tags,
      [t.strip().split() for t in targets],
      [w.strip().split() for w in words],
      output_dir + "tagging.result"
    )
  pass