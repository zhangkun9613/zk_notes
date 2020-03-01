import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np

class Model(object):
  def __init__(self, hparams, iterator, mode, vocabs, reverse_dst_table):
    self.hparams = hparams
    self.iterator = iterator
    self.mode = mode
    self.vocabs = vocabs
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    elif mode == tf.contrib.learn.ModeKeys.EVAL:
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    else:
      self.dropout_keep_prob = tf.constant(1.0, dtype=tf.float32)
    scores, predictions = self.build_graph(hparams)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      weights = tf.reduce_sum(hparams.class_weights * self.iterator.dst_output, axis=1)
      loss = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.iterator.dst_output)
      loss = tf.reduce_mean(loss * weights)
      correct_predictions = tf.equal(predictions, tf.argmax(self.iterator.dst_output, 1))
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


  def train(self, sess):
    return sess.run([self.update, self.loss, self.accuracy, self.train_summary, self.global_step],
                    {self.dropout_keep_prob: self.hparams.train_dropout_keep_prob})


  def eval(self, sess):
    return sess.run([self.predictions],
                    {self.dropout_keep_prob: 1.0})


  def build_graph(self, hparams):
    h_pool = self.build_cnn(hparams, self.dropout_keep_prob)
    output = self.build_rnn(hparams, self.dropout_keep_prob, h_pool)
    with tf.name_scope("output"):
      scores = tf.layers.dense(inputs=output, units=self.vocabs.dst_vocab_size, activation=None)
      predictions = tf.argmax(scores, 1, name="predictions")
    return scores, predictions


  def build_cnn(self, hparams, dropout_keep_prob):
    word_embedding = tf.get_variable('word_embedding',
                                     [self.vocabs.word_vocab_size, hparams.word_embedding_size], tf.float32)
    word_embedded = tf.nn.embedding_lookup(word_embedding, self.iterator.word_input)
    pos_embedding = tf.get_variable('pos_embedding',
                                    [self.vocabs.pos_vocab_size, hparams.pos_embedding_size], tf.float32)
    pos_embedded = tf.nn.embedding_lookup(pos_embedding, self.iterator.pos_input)
    embedded_input = tf.concat([word_embedded, pos_embedded], -1)
    embedded_input_expand = tf.expand_dims(embedded_input, -1)
    embedded_input_expand = tf.transpose(embedded_input_expand, [0, 2, 1, 3])
    embedding_size = hparams.word_embedding_size + hparams.pos_embedding_size
    pooled_outputs = []
    reduced = np.int32(np.ceil(hparams.max_seq_len * 1.0 / hparams.max_pool_size))
    for i, filter_size in enumerate(hparams.filter_sizes):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
        conv = tf.layers.conv2d(inputs=embedded_input_expand, filters=hparams.num_filters,
                                kernel_size=[embedding_size, filter_size], strides=[embedding_size, 1],
                                activation=tf.nn.relu, padding='same')
        pooled = tf.layers.max_pooling2d(
          inputs=conv, pool_size=[1, hparams.max_pool_size], strides=[1, hparams.max_pool_size])
        pooled = tf.reshape(pooled, [-1, reduced, hparams.num_filters])
        pooled_outputs.append(pooled)

    h_pool = tf.concat(pooled_outputs, 2)
    return tf.nn.dropout(h_pool, dropout_keep_prob)


  def build_rnn(self, hparams, dropout_keep_prob, h_pool):
    reduced = np.int32(np.ceil(hparams.max_seq_len * 1.0 / hparams.max_pool_size))
    cell = rnn.GRUCell(num_units=hparams.rnn_size)
    cell = rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    inputs = [tf.squeeze(input_, [1]) for input_ in tf.split(h_pool, int(reduced), axis=1)]
    outputs, state = tf.nn.static_rnn(cell, inputs, sequence_length=self.iterator.src_seq_len, dtype=tf.float32)
    output = outputs[0]
    with tf.variable_scope('rnn-output'):
      tf.get_variable_scope().reuse_variables()
      one = tf.ones([1, hparams.rnn_size], tf.float32)
      for i in range(1, len(outputs)):
        ind = self.iterator.src_seq_len < (i + 1) * hparams.max_pool_size
        ind = tf.to_float(ind)
        ind = tf.expand_dims(ind, -1)
        mat = tf.matmul(ind, one)
        output = tf.add(tf.multiply(output, mat), tf.multiply(outputs[i], 1.0 - mat))
    return output
