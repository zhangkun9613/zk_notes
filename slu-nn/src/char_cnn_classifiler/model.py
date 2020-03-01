import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


class Model(object):
  def __init__(self, hparams, iterator, vocabs, mode, reverse_dst_table):
    self.hparams = hparams
    self.iterator = iterator
    self.mode = mode
    self.vocabs = vocabs
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.dropout_keep_prob = hparams.dropout_keep_prob
    else:
      self.dropout_keep_prob = 1.0
    self.l2_loss = tf.constant(0.0)
    scores, predictions = self.build_graph(hparams)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      # weights = tf.reduce_sum(hparams.class_weights * self.iterator.target_input, axis=1)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=self.iterator.target_input)
      loss = tf.reduce_mean(loss) + 0.0 * self.l2_loss
      correct_predictions = tf.equal(tf.to_int32(predictions), self.iterator.target_input)
      accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
      self.loss = loss
      self.accuracy = accuracy
    else:
      self.predictions = reverse_dst_table.lookup(tf.to_int64(predictions))

    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(hparams.learning_rate)
      grads_and_vars = optimizer.compute_gradients(self.loss)
      self.update = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
      loss_summary = tf.summary.scalar("loss", self.loss)
      acc_summary = tf.summary.scalar("accuracy", self.accuracy)
      self.train_summary = tf.summary.merge([loss_summary, acc_summary])

    self.saver = tf.train.Saver(tf.global_variables())

  def train(self, sess, batch):
    char_input, word_input, pos_input, word_len, char_len, target_input = zip(*batch)
    return sess.run([self.update, self.loss, self.accuracy, self.train_summary, self.global_step],
                    {
                      self.iterator.pos_input: pos_input,
                      self.iterator.word_input: word_input,
                      self.iterator.char_input: char_input,
                      self.iterator.word_len: word_len,
                      self.iterator.char_len: char_len,
                      self.iterator.target_input: target_input
                    })

  def eval(self, sess, batch):
    char_input, word_input, pos_input, word_len, char_len, target_input = zip(*batch)
    return sess.run([self.predictions],
                    {
                      self.iterator.pos_input: pos_input,
                      self.iterator.word_input: word_input,
                      self.iterator.char_input: char_input,
                      self.iterator.word_len: word_len,
                      self.iterator.char_len: char_len,
                      self.iterator.target_input: target_input
                    })

  def build_graph(self, hparams):
    encoded_chars = self.char_encoder()
    encoder_output, encoder_state = self.word_encoder(encoded_chars)
    output = self.attention_layer(encoder_output)
    return self.get_logits(output)
    # h_pool = self.build_cnn(hparams, self.dropout_keep_prob)
    # output = self.build_rnn(hparams, self.dropout_keep_prob, h_pool)
    # with tf.name_scope("output"):
    #   scores = tf.layers.dense(inputs=output, units=self.vocabs.target_vocab_size, activation=None)
    #   predictions = tf.argmax(scores, 1, name="predictions")
    # return scores, predictions

  def char_encoder(self):
    with tf.variable_scope("char_encoder"):
      batch_size = tf.shape(self.iterator.char_input)[0]
      chars = tf.transpose(
        tf.reshape(self.iterator.char_input, [-1, self.hparams.max_char_len]))
      char_embedding = tf.get_variable(
        'char_encoder_embedding', [self.vocabs.char_vocab_size, self.hparams.char_embedding_size], tf.float32)
      chars_inputs = tf.nn.embedding_lookup(char_embedding, chars)
      chars_len = tf.reshape(self.iterator.char_len, [-1])
      # chars_len = tf.concat(self.iterator.char_len, 0)
      # self.char_embedding = chars_inputs
      fwcell = self.create_cell(self.dropout_keep_prob)
      bwcell = self.create_cell(self.dropout_keep_prob)
      _, encoder_states = tf.nn.bidirectional_dynamic_rnn(
        fwcell, bwcell, chars_inputs, dtype=tf.float32, sequence_length=chars_len, time_major=True)
      output_state_fw, output_state_bw = encoder_states
      encoder_states = tf.concat([output_state_fw.c, output_state_bw.c], -1)
      # print(encoder_states.get_shape().as_list())
      # print(fwcell.output_size)
      encoder_states = tf.reshape(
        encoder_states,
        [batch_size, -1, self.hparams.num_hidden_units * 2])
      return encoder_states

  def word_encoder(self, encoded_chars):
    with tf.variable_scope("word_encoder"):
      words = tf.transpose(self.iterator.word_input)
      pos = tf.transpose(self.iterator.pos_input)
      encoded_chars = tf.transpose(encoded_chars, [1, 0, 2])
      # print(encoded_chars.get_shape().as_list())
      # TODO: add words vectors
      # word_vectors = tf.transpose(self.iterator.word_vectors, [1, 0, 2])
      word_embedding = tf.get_variable(
        'word_encoder_embedding', [self.vocabs.word_vocab_size, self.hparams.word_embedding_size], tf.float32)
      pos_embedding = tf.get_variable(
        'pos_encoder_embedding', [self.vocabs.pos_vocab_size, 50], tf.float32)
      word_inputs = tf.nn.embedding_lookup(word_embedding, words)
      pos_inputs = tf.nn.embedding_lookup(pos_embedding, pos)
      inputs = tf.concat([word_inputs, encoded_chars, pos_inputs], -1)
      #inputs = tf.concat([word_inputs, pos_inputs], -1)
      #inputs = tf.concat([word_inputs], -1)
      # inputs = word_vectors
      # self.embedding = word_inputs
      fwcell = self.create_cell(self.dropout_keep_prob)
      bwcell = self.create_cell(self.dropout_keep_prob)
      encoder_outputs, encoder_states = tf.nn.bidirectional_dynamic_rnn(
        fwcell, bwcell, inputs, dtype=tf.float32, sequence_length=self.iterator.word_len, time_major=True)
      return encoder_outputs, encoder_states

  def attention_layer(self, outputs):
    batch_size = tf.shape(outputs[0])[1]
    W = tf.Variable(tf.random_normal([self.hparams.num_hidden_units], stddev=0.1))
    H = tf.transpose(outputs[0], [1, 0, 2]) + tf.transpose(outputs[1], [1, 0,2])
    M = tf.tanh(H)
    alpha = tf.nn.softmax(tf.matmul(tf.reshape(M, [-1, self.hparams.num_hidden_units]), tf.reshape(W, [-1, 1])))
    r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [batch_size, -1, 1]))
    r = tf.squeeze(r, [2])
    h_star = tf.tanh(r)
    drop = tf.nn.dropout(h_star, self.dropout_keep_prob)
    return drop 

  def get_logits(self, encoder_states):
    #output_state_fw, output_state_bw = encoder_states
    #encoder_states = tf.concat([output_state_fw.c, output_state_bw.c], -1)
    softmax_w = tf.get_variable(
      "softmax_w",
      shape=[self.hparams.num_hidden_units, self.vocabs.target_vocab_size],
      dtype=tf.float32)
    softmax_b = tf.get_variable(
      "softmax_b",
      shape=[self.vocabs.target_vocab_size],
      dtype=tf.float32)
    self.l2_loss += tf.nn.l2_loss(softmax_w)
    self.l2_loss += tf.nn.l2_loss(softmax_b)
    logits = tf.nn.xw_plus_b(encoder_states, softmax_w, softmax_b)
    predictions = tf.argmax(logits, 1, name="predictions")
    return logits, predictions

  def create_cell(self, dropout_keep, layers=1):
    cells = []
    for i in range(layers):
      cell = rnn.BasicLSTMCell(self.hparams.num_hidden_units, forget_bias=1.0)
      if dropout_keep < 1.0:
        cell = rnn.DropoutWrapper(cell, dropout_keep, dropout_keep)
      cells.append(cell)
    if len(cells) == 1:
      return cells[0]
    else:
      return rnn.MultiRNNCell(cells, state_is_tuple=True)

