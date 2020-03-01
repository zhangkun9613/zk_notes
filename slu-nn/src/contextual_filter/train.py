import tensorflow as tf
import collections
from src.contextual_filter import utils
from src.contextual_filter import model

class Vocabs(
  collections.namedtuple(
    "Vocabs", ("word_vocab_size", "pos_vocab_size", "tag_vocab_size", "help_vocab_size",
    "word_vocab", "pos_vocab", "tag_vocab", "help_vocab", "rev_help_vocab"))):
  pass

class TrainModel(
  collections.namedtuple("TrainModel", ("graph", "model", "history", "current"))):
  pass

class ServingModel(
  collections.namedtuple("ServingModel", (
      "graph", "model", "history", "current",
      "c_word_placeholder", "c_pos_placeholder", "c_tag_placeholder", "c_wv_placeholder",
      "h_word_placeholder", "h_pos_placeholder", "h_wv_placeholder", "h_tag_placeholder"))):
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
  vocabs = 'vocabs/{0}.vocab'
  word_vocab, _ = utils.initialize_vocab(input_dir + vocabs.format("word"))
  pos_vocab, _ = utils.initialize_vocab(input_dir + vocabs.format("pos"))
  tag_vocab, _ = utils.initialize_vocab(input_dir + vocabs.format("tag"))
  help_vocab, rev_help_vocab = utils.initialize_vocab(input_dir + vocabs.format("help"))
  return Vocabs(
    word_vocab = word_vocab,
    word_vocab_size= len(word_vocab),
    pos_vocab = pos_vocab,
    pos_vocab_size = len(pos_vocab),
    tag_vocab = tag_vocab,
    tag_vocab_size = len(tag_vocab),
    help_vocab = help_vocab,
    rev_help_vocab = rev_help_vocab,
    help_vocab_size = len(help_vocab))


def create_train_model(hparams, input_dir):
  graph = tf.Graph()
  with graph.as_default():
    vocabs = create_vocabs(input_dir)
    history = utils.History(
      word_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      pos_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      word_vector=tf.placeholder(tf.float32, [None, hparams.max_seq_len, hparams.word_embedding_size]),
      tag_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      help=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      word_len=tf.placeholder(tf.int32, [None]),
      help_len=tf.placeholder(tf.int32, [None]),
      batch_size=hparams.batch_size)
    current = utils.Current(
      word_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      pos_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      word_vector=tf.placeholder(tf.float32, [None, hparams.max_seq_len, hparams.word_embedding_size]),
      tag_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      word_len=tf.placeholder(tf.int32, [None]),
      batch_size=hparams.batch_size)
    m = model.Model(hparams, vocabs, history, current, tf.contrib.learn.ModeKeys.TRAIN, None)
    return TrainModel(graph=graph, model=m, history=history, current=current)

def create_eval_model(hparams, input_dir):
  graph = tf.Graph()
  with graph.as_default():
    vocabs = create_vocabs(input_dir)
    history = utils.History(
      word_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      pos_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      word_vector=tf.placeholder(tf.float32, [None, hparams.max_seq_len, hparams.word_embedding_size]),
      tag_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      help=None,
      word_len=tf.placeholder(tf.int32, [None]),
      help_len=None,
      batch_size=1)
    current = utils.Current(
      word_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      pos_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      word_vector=tf.placeholder(tf.float32, [None, hparams.max_seq_len, hparams.word_embedding_size]),
      tag_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      word_len=tf.placeholder(tf.int32, [None]),
      batch_size=1)
    rev_help_table = tf.contrib.lookup.index_to_string_table_from_file(
      input_dir + "vocabs/help.vocab", default_value='<unk>')
    m = model.Model(hparams, vocabs, history, current, tf.contrib.learn.ModeKeys.EVAL, rev_help_table)
    return TrainModel(graph=graph, history=history, current=current, model=m)

def create_serving_model(hparams, input_dir):
  graph = tf.Graph()
  with graph.as_default():
    vocabs = create_vocabs(input_dir)
    word_vocab_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(list(vocabs.word_vocab.keys()),
                                                  list(vocabs.word_vocab.values())),
      0)
    pos_vocab_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(list(vocabs.pos_vocab.keys()),
                                                  list(vocabs.pos_vocab.values())),
      0)
    tag_vocab_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(list(vocabs.tag_vocab.keys()),
                                                  list(vocabs.tag_vocab.values())),
      0)
    rev_help_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(list(vocabs.help_vocab.values()),
                                                  list(vocabs.help_vocab.keys()),
                                                  key_dtype=tf.int64,
                                                  value_dtype=tf.string),
      'N')
    c_word_place_holder = tf.placeholder(shape=[], dtype=tf.string)
    c_pos_place_holder = tf.placeholder(shape=[], dtype=tf.string)
    c_tag_place_holder = tf.placeholder(shape=[], dtype=tf.string)
    c_wv_place_holder = tf.placeholder(shape=[None, hparams.word_embedding_size], dtype=tf.float32)
    h_word_place_holder = tf.placeholder(shape=[], dtype=tf.string)
    h_pos_place_holder = tf.placeholder(shape=[], dtype=tf.string)
    h_tag_place_holder = tf.placeholder(shape=[], dtype=tf.string)
    h_wv_place_holder = tf.placeholder(shape=[None, hparams.word_embedding_size], dtype=tf.float32)
    current, history = utils.get_serving_input(
      c_word_place_holder, c_pos_place_holder, c_tag_place_holder, c_wv_place_holder,
      h_word_place_holder, h_pos_place_holder, h_tag_place_holder, h_wv_place_holder,
      word_vocab_table, pos_vocab_table, tag_vocab_table, hparams.max_seq_len, hparams.fasttext_model)
    m = model.Model(hparams, vocabs, history, current, tf.contrib.learn.ModeKeys.INFER, rev_help_table)
    return ServingModel(graph=graph, history=history, current=current, model=m,
                        c_word_placeholder=c_word_place_holder,
                        c_pos_placeholder=c_pos_place_holder,
                        c_tag_placeholder=c_tag_place_holder,
                        c_wv_placeholder=c_wv_place_holder,
                        h_word_placeholder=h_word_place_holder,
                        h_pos_placeholder=h_pos_place_holder,
                        h_wv_placeholder=h_wv_place_holder,
                        h_tag_placeholder=h_tag_place_holder)

def train(hparams, input_dir, output_dir):
  history, current, _, _, _, _ = utils.load_data(
    hparams.fasttext_model, input_dir, hparams.max_seq_len)
  train_model = create_train_model(hparams, input_dir)
  with tf.Session(graph=train_model.graph) as sess:
    summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())
    global_step = train_model.model.global_step.eval(session=sess)
    batch_num = int(hparams.max_steps/(len(history)/hparams.batch_size))
    batches = utils.get_batch(current, history, hparams.batch_size, batch_num)
    for cur_step, batch in enumerate(batches):
      pre_batch, cur_batch = batch
      if cur_step % 50 == 0: print("cur_step:  ", cur_step)
      (_, _, step_summary, global_step) = train_model.model.train(sess, pre_batch, cur_batch)
      summary_writer.add_summary(step_summary, global_step)
    train_model.model.saver.save(sess, output_dir + 'model.ckpt', global_step=global_step)
    summary_writer.close()

def eval(hparams, input_dir, output_dir, prefix):
  history, current, _, _, _, _ = utils.load_data(
    hparams.fasttext_model, input_dir, hparams.max_seq_len, prefix + ".src")
  eval_model = create_eval_model(hparams, input_dir)
  with tf.Session(graph=eval_model.graph) as sess:
    ckpt = tf.train.latest_checkpoint(output_dir)
    eval_model.model.saver.restore(sess, ckpt)
    sess.run(tf.tables_initializer())
    batches = utils.get_batch(current, history, len(history), 1, shuffle=False)
    for batch in batches:
      pre_batch, cur_batch = batch
      (predictions, probabilities) = eval_model.model.eval(sess, pre_batch, cur_batch)
    help = []
    # print predictions
    for i in range(len(predictions)):
      prediction = predictions[i].tolist()
      try:
        help_len = prediction.index(b'</s>')
        prediction = prediction[:help_len]
      except ValueError:
        print ("Error incurs")
        pass
      help.append([l.decode('utf-8') for l in prediction])
    # print help
    with tf.gfile.GFile(input_dir + prefix + ".src", mode='r') as f:
      data = [d.split("\t") for d in f.read().splitlines()]
      words = [d[0] for d in data]
      targets = [d[6] for d in data]
      conlleval(
        help,
        [t.split() for t in targets],
        [w.split() for w in words],
        output_dir + "help_result")

def export(hparams, input_dir, output_dir, export_dir):
  serve_model = create_serving_model(hparams, input_dir)
  with tf.Session(graph=serve_model.graph) as sess:
    ckpt = tf.train.latest_checkpoint(output_dir)
    serve_model.model.saver.restore(sess, ckpt)
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    c_word_placeholder_info = tf.saved_model.utils.build_tensor_info(serve_model.c_word_placeholder)
    c_pos_placeholder_info = tf.saved_model.utils.build_tensor_info(serve_model.c_pos_placeholder)
    c_tag_placeholder_info = tf.saved_model.utils.build_tensor_info(serve_model.c_tag_placeholder)
    c_wv_placeholder_info = tf.saved_model.utils.build_tensor_info(serve_model.c_wv_placeholder)
    h_word_placeholder_info = tf.saved_model.utils.build_tensor_info(serve_model.h_word_placeholder)
    h_pos_placeholder_info = tf.saved_model.utils.build_tensor_info(serve_model.h_pos_placeholder)
    h_tag_placeholder_info = tf.saved_model.utils.build_tensor_info(serve_model.h_tag_placeholder)
    h_wv_placeholder_info = tf.saved_model.utils.build_tensor_info(serve_model.h_wv_placeholder)
    output_info = tf.saved_model.utils.build_tensor_info(serve_model.model.predictions)
    scores_info = tf.saved_model.utils.build_tensor_info(serve_model.model.probabilities)
    prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
          'c_words': c_word_placeholder_info,
          'c_pos': c_pos_placeholder_info,
          'c_tag': c_tag_placeholder_info,
          'c_wv': c_wv_placeholder_info,
          'h_words': h_word_placeholder_info,
          'h_pos': h_pos_placeholder_info,
          'h_tag': h_tag_placeholder_info,
          'h_wv': h_wv_placeholder_info,
        },
        outputs={
          'outputs': output_info,
          'scores': scores_info,
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
      )
    )
    init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
        'contextual_filter': prediction_signature
      },
      legacy_init_op=init_op)
    builder.save()
