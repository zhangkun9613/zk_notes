import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import fully_connected

class Model(object):
  def __init__(self, hparams, vocabs, history, current, mode, rev_help_table):
    self.hparams = hparams
    self.vocabs = vocabs
    self.history = history
    self.current = current
    self.global_step = tf.Variable(0, trainable=False)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.dropout_keep_prob = hparams.dropout_keep_prob
    else:
      self.dropout_keep_prob = 1.0
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      logits, _ = self.build_graph()
      labels = tf.transpose(history.help)
      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
      output_weight = tf.sequence_mask(history.help_len, hparams.max_seq_len, dtype=tf.float32)
      output_weight = tf.transpose(output_weight)
      self.loss = tf.reduce_sum(crossent * output_weight) / tf.to_float(history.batch_size)
      params = tf.trainable_variables()
      opt = tf.train.AdamOptimizer(hparams.learning_rate)
      gradients = tf.gradients(self.loss, params, colocate_gradients_with_ops=True)
      clipped_gradients, gradients_norm = tf.clip_by_global_norm(gradients, 5.0)
      self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
      self.train_summary = tf.summary.merge([tf.summary.scalar('train_loss', self.loss)])
    else:
      logits, predictions = self.build_graph()
      predictions = tf.transpose(predictions)
      probabilities = tf.transpose(tf.nn.softmax(logits), [1, 0, 2])
      if mode == tf.contrib.learn.ModeKeys.INFER:
        predictions = tf.slice(predictions, [0, 0], [-1, self.history.word_len[0]])
        probabilities = tf.slice(probabilities, [0, 0, 0], [-1, self.history.word_len[0], -1])
        probabilities = tf.reduce_max(probabilities, -1)
      self.probabilities = probabilities
      self.predictions = rev_help_table.lookup(predictions)
    self.saver = tf.train.Saver(tf.global_variables())

  def eval(self, sess, pre_batch, cur_batch):
    pre_words, pre_word_vectors, pre_pos, pre_tags, pre_words_len, _, _ = zip(*pre_batch)
    words, word_vectors, pos, tags, words_len = zip(*cur_batch)
    return sess.run(
      [self.predictions, self.probabilities],
      {
        self.history.word_input: pre_words,
        self.history.word_vector: pre_word_vectors,
        self.history.word_len: pre_words_len,
        self.history.pos_input: pre_pos,
        self.history.tag_input: pre_tags,
        self.current.tag_input: tags,
        self.current.pos_input: pos,
        self.current.word_input: words,
        self.current.word_vector: word_vectors,
        self.current.word_len: words_len,
      })


  def train(self, sess, pre_batch, cur_batch):
    pre_words, pre_word_vectors, pre_pos, pre_tags, pre_words_len, help, help_len = zip(*pre_batch)
    words, word_vectors, pos, tags, words_len = zip(*cur_batch)
    return sess.run(
      [self.update, self.loss, self.train_summary, self.global_step],
      {
        self.history.word_input: pre_words,
        self.history.word_vector: pre_word_vectors,
        self.history.word_len: pre_words_len,
        self.history.pos_input: pre_pos,
        self.history.tag_input: pre_tags,
        self.history.help: help,
        self.history.help_len: help_len,
        self.current.tag_input: tags,
        self.current.pos_input: pos,
        self.current.word_input: words,
        self.current.word_vector: word_vectors,
        self.current.word_len: words_len,
      })

  def get_knowledge(self, pre_inputs, inputs):
    with tf.variable_scope("memory"):
      memory, _ = self.create_encoder(inputs, self.history.word_len)  # memory only use rnn
    with tf.variable_scope("tagger_encoder"):
      _, encoder_states = self.create_encoder(pre_inputs, self.current.word_len)
    memory = tf.transpose(memory, [1, 0, 2])
    encoder_states_expended = tf.expand_dims(encoder_states, -1)
    attention = tf.nn.softmax(tf.matmul(memory, encoder_states_expended), 1)
    context = tf.reduce_sum(tf.multiply(attention, memory), 1)
    context = context + encoder_states
    # print context.get_shape().as_list()
    knowledge = fully_connected(context, self.hparams.num_hidden_units)
    # print knowledge.get_shape().as_list()
    return knowledge

  def build_graph(self):
    pre_inputs, inputs = self.create_embedding()
    knowledge = self.get_knowledge(pre_inputs, inputs)
    return self.RNN_tagger(knowledge, pre_inputs)

  def create_cell(self, dropout_keep, layers=1):
    cells = []
    for _ in range(layers):
      cell = rnn.GRUCell(self.hparams.num_hidden_units) # GRU or LSTM need deciding
      if dropout_keep < 1.0:
        cell = rnn.DropoutWrapper(cell, dropout_keep, dropout_keep)
      cells.append(cell)
    if len(cells) == 1: return cells[0]
    else: return rnn.MultiRNNCell(cells, state_is_tuple=True)

  def create_embedding(self):
    pre_word_input = tf.transpose(self.history.word_input)
    word_input = tf.transpose(self.current.word_input)
    pre_pos_input = tf.transpose(self.history.pos_input)
    pos_input = tf.transpose(self.current.pos_input)
    pre_tag_input = tf.transpose(self.history.tag_input)
    tag_input = tf.transpose(self.current.tag_input)
    pre_wv_input = tf.transpose(self.history.word_vector, [1, 0, 2])
    wv_input = tf.transpose(self.current.word_vector, [1, 0, 2])
    word_embedding = tf.get_variable(
      'word_embedding', [self.vocabs.word_vocab_size, self.hparams.word_embedding_size], tf.float32)
    pos_embedding = tf.get_variable(
      'pos_embedding', [self.vocabs.pos_vocab_size, self.hparams.word_embedding_size], tf.float32)
    tag_embedding = tf.get_variable(
      'tag_embedding', [self.vocabs.tag_vocab_size, self.hparams.word_embedding_size], tf.float32)
    pre_word_input = tf.nn.embedding_lookup(word_embedding, pre_word_input)
    pre_pos_input = tf.nn.embedding_lookup(pos_embedding, pre_pos_input)
    pre_tag_input = tf.nn.embedding_lookup(tag_embedding, pre_tag_input)
    word_input = tf.nn.embedding_lookup(word_embedding, word_input)
    pos_input = tf.nn.embedding_lookup(pos_embedding, pos_input)
    tag_input = tf.nn.embedding_lookup(tag_embedding, tag_input)
    pre_inputs = tf.concat([pre_word_input, pre_pos_input, pre_tag_input, pre_wv_input], -1)
    inputs = tf.concat([word_input, pos_input, tag_input, wv_input], -1)
    return pre_inputs, inputs

  def create_encoder(self, inputs, src_len):
    fwcell = self.create_cell(self.hparams.dropout_keep_prob)
    bwcell = self.create_cell(self.hparams.dropout_keep_prob)
    encoder_outputs, encoder_states = tf.nn.bidirectional_dynamic_rnn(
      fwcell, bwcell, inputs, dtype=tf.float32, sequence_length=src_len, time_major=True)
    return tf.concat(encoder_outputs, -1), tf.concat(encoder_states, -1)

  def RNN_tagger(self, knowledge, pre_inputs):
    cell = self.create_cell(self.dropout_keep_prob)
    knowledge = tf.tile(knowledge, [self.hparams.max_seq_len, 1])
    knowledge = tf.reshape(
      knowledge,
      [self.hparams.max_seq_len, -1, self.hparams.num_hidden_units])
    inputs = tf.concat([knowledge, pre_inputs], -1)
    outputs, _ = tf.nn.dynamic_rnn(
      cell, inputs, dtype=tf.float32, sequence_length=self.history.word_len, time_major=True)
    outputs = tf.unstack(outputs)
    softmax_w = tf.get_variable(
      "softmax_w",
      shape=[self.hparams.num_hidden_units, self.vocabs.help_vocab_size],
      dtype=tf.float32)
    softmax_b = tf.get_variable(
      "softmax_b",
      shape=[self.vocabs.help_vocab_size],
      dtype=tf.float32)
    logits = [tf.nn.xw_plus_b(output, softmax_w, softmax_b) for output in outputs]
    predictions = tf.argmax(logits, 2, name="predictions", output_type=tf.int64)
    return logits, predictions
