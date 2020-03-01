import tensorflow as tf
import numpy as np
from tensorflow.python.ops import lookup_ops
import collections
from src.char_cnn_classifiler import iterator
from src.char_cnn_classifiler import model

word_vocab_path = "vocabs/word.vocab"
pos_vocab_path = "vocabs/pos.vocab"
char_vocab_path = "vocabs/char.vocab"
target_vocab_path = "vocabs/target.vocab"

class TrainModel(
  collections.namedtuple("TrainModel", ("graph", "iterator", "model"))):
  pass

class ServingModel(
  collections.namedtuple("ServingModel", ("graph", "model", "iterator", "char_placeholder",
                                          "pos_placeholder", "word_placeholder", "char_len_placeholder"))):
  pass

class Vocabs(
  collections.namedtuple("Vocabs", (
      "char_vocab", "char_vocab_size",
      "word_vocab", "word_vocab_size",
      "pos_vocab", "pos_vocab_size",
      "reverse_target_vocab", "target_vocab", "target_vocab_size"))):
  pass


def create_vocabs(input_dir):
  char_vocab, _ = iterator.initialize_vocabulary(input_dir + char_vocab_path)
  pos_vocab, _ = iterator.initialize_vocabulary(input_dir + pos_vocab_path)
  word_vocab, _ = iterator.initialize_vocabulary(input_dir + word_vocab_path)
  target_vocab, reverse_target_vocab = iterator.initialize_vocabulary(input_dir + target_vocab_path)
  return Vocabs(char_vocab=char_vocab,
                char_vocab_size=len(char_vocab),
                word_vocab=word_vocab,
                word_vocab_size=len(word_vocab),
                pos_vocab=pos_vocab,
                pos_vocab_size=len(pos_vocab),
                target_vocab=target_vocab,
                reverse_target_vocab=reverse_target_vocab,
                target_vocab_size=len(target_vocab))

def create_eval_model(hparams, input_dir):
  graph = tf.Graph()
  with graph.as_default():
    vocabs = create_vocabs(input_dir)
    it = iterator.Iterator(
      word_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      pos_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      char_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len, hparams.max_char_len]),
      target_input=tf.placeholder(tf.int32, [None]),
      word_len=tf.placeholder(tf.int32, [None]),
      char_len=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      batch_size=hparams.batch_size)
    reverse_taget_table = lookup_ops.index_to_string_table_from_file(
      input_dir + target_vocab_path, default_value='<unk>')
    m = model.Model(hparams, it, vocabs, tf.contrib.learn.ModeKeys.EVAL, reverse_taget_table)
    return TrainModel(
      graph=graph,
      model=m,
      iterator=it)

def create_serve_model(hparams, input_dir):
  graph = tf.Graph()
  with graph.as_default():
    vocabs = create_vocabs(input_dir)
    char_vocab_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        list(vocabs.char_vocab.keys()),
        list(vocabs.char_vocab.values()),
      ), vocabs.char_vocab["<unk>"]
    )
    pos_vocab_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        list(vocabs.pos_vocab.keys()),
        list(vocabs.pos_vocab.values()),
      ), vocabs.pos_vocab["<unk>"]
    )
    word_vocab_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        list(vocabs.word_vocab.keys()),
        list(vocabs.word_vocab.values()),
      ), vocabs.word_vocab["<unk>"]
    )
    target_vocab_table = tf.contrib.lookup.HashTable(
      tf.contrib.lookup.KeyValueTensorInitializer(
        list(vocabs.target_vocab.values()),
        list(vocabs.target_vocab.keys()),
        key_dtype=tf.int64,
        value_dtype=tf.string,
      ), "unknown"
    )
    char_place_holder = tf.placeholder(shape=[None, hparams.max_char_len], dtype=tf.string)
    pos_place_holder = tf.placeholder(shape=[None], dtype=tf.string)
    word_place_holder = tf.placeholder(shape=[None], dtype=tf.string)
    char_len_place_holder = tf.placeholder(shape=[None], dtype=tf.int32)
    it = iterator.get_serving_iterator(
      char_len_place_holder,
      char_place_holder, char_vocab_table,
      pos_place_holder, pos_vocab_table,
      word_place_holder, word_vocab_table,
      hparams.max_seq_len)
    m = model.Model(hparams, it, vocabs, tf.contrib.learn.ModeKeys.INFER, target_vocab_table)
    return ServingModel(
      graph=graph,
      model=m,
      iterator=it,
      char_placeholder=char_place_holder,
      pos_placeholder=pos_place_holder,
      word_placeholder=word_place_holder,
      char_len_placeholder=char_len_place_holder,
    )

def create_train_model(hparams, input_dir):
  graph = tf.Graph()
  with graph.as_default():
    vocabs = create_vocabs(input_dir)
    it = iterator.Iterator(
      word_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      pos_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      char_input=tf.placeholder(tf.int32, [None, hparams.max_seq_len, hparams.max_char_len]),
      target_input=tf.placeholder(tf.int32, [None]),
      word_len=tf.placeholder(tf.int32, [None]),
      char_len=tf.placeholder(tf.int32, [None, hparams.max_seq_len]),
      batch_size=hparams.batch_size)
    m = model.Model(hparams, it, vocabs, tf.contrib.learn.ModeKeys.TRAIN, None)
  return TrainModel(
    graph=graph,
    model=m,
    iterator=it)

def train(hparams, input_dir, output_dir):
  data = iterator.load_data(
    hparams.fasttext_model, input_dir, hparams.max_char_len, hparams.max_seq_len, tf.contrib.learn.ModeKeys.TRAIN)
  train_model = create_train_model(hparams, input_dir)
  with tf.Session(graph=train_model.graph) as sess:
    summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
    #ckpt = tf.train.latest_checkpoint(output_dir)
    #train_model.model.saver.restore(sess, ckpt)
    sess.run(tf.global_variables_initializer())
    global_step = train_model.model.global_step.eval(session=sess)
    batch_num = int(hparams.num_train_steps / (len(data) / hparams.batch_size))
    batches = iterator.get_batch(data, hparams.batch_size, batch_num)
    for cur_step, batch in enumerate(batches):
      #print([(np.asarray(b[0]).shape, np.asarray(b[1]).shape, np.asarray(b[2]).shape) for b in batch])
      (_, loss, accuracy, step_summary, global_step) = train_model.model.train(sess, batch)
      if cur_step > 0 and cur_step % 500 == 0:
        train_model.model.saver.save(sess, output_dir + 'model.ckpt', global_step=global_step)
        print(cur_step, loss, accuracy)
        eval(hparams, input_dir, output_dir)
      # print (train_model.model.logging)
      summary_writer.add_summary(step_summary, global_step)
    summary_writer.close()

def eval(hparams, input_dir, output_dir):
  with tf.gfile.GFile(input_dir + 'dev.in', mode='r') as f:
    data = f.read().splitlines()
    target = []
    for i in range(0, len(data)-1, 3):
      target.append((data[i], data[i+2]))
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
      result = [r.decode('utf-8') for r in predictions]
      count = 0
      print(len(target),len(result))
      all_result =  []
      for i in range(len(target)):
        if target[i][1] != result[i]:
          count = count + 1
          print(target[i],result[i])
        all_result.append(target[i][0] +'\t'+ target[i][1] +'\t'+ result[i])
      print((len(target) - count) / len(target))
    with open(output_dir + "result.txt", "w") as cout:
      #cout.write('\n'.join(map(lambda x: x.decode('utf-8'), predictions.tolist())))
      cout.write('\n'.join(all_result))

def export(hparams, input_dir, output_dir, export_dir):
  server_model = create_serve_model(hparams, input_dir)
  with tf.Session(graph=server_model.graph) as sess:
    ckpt = tf.train.latest_checkpoint(output_dir)
    server_model.model.saver.restore(sess, ckpt)
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    char_placeholder_info = tf.saved_model.utils.build_tensor_info(server_model.char_placeholder)
    word_placeholder_info = tf.saved_model.utils.build_tensor_info(server_model.word_placeholder)
    pos_placeholder_info = tf.saved_model.utils.build_tensor_info(server_model.pos_placeholder)
    char_len_placeholder_info = tf.saved_model.utils.build_tensor_info(server_model.char_len_placeholder)
    outputs_info = tf.saved_model.utils.build_tensor_info(server_model.model.predictions)
    predictions_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
          "char": char_placeholder_info,
          "word": word_placeholder_info,
          "pos": pos_placeholder_info,
          "char_len": char_len_placeholder_info
        },
        outputs={
          "outputs": outputs_info
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
      )
    )
    init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
        "classification": predictions_signature
      },
      legacy_init_op=init_op
    )
    builder.save()
