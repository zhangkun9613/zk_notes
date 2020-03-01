import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core


class Model(object):
  def __init__(self, hparams, vocabs, iterator, mode, reverse_target_table):
    self.hparams = hparams
    self.vocabs = vocabs
    self.iterator = iterator
    self.global_step = tf.Variable(0, trainable=False)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.dropout_keep_prob = hparams.dropout_keep_prob
      logits, _ = self.build_graph(mode)
      labels = tf.transpose(iterator.target_out)
      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
      output_weights = tf.sequence_mask(iterator.word_len, hparams.max_seq_len + 2, dtype=logits.dtype)
      output_weights = tf.transpose(output_weights)
      self.loss = tf.reduce_sum(crossent * output_weights) / tf.to_float(iterator.batch_size)
      params = tf.trainable_variables()
      opt = tf.train.AdamOptimizer(hparams.learning_rate)
      gradients = tf.gradients(self.loss, params, colocate_gradients_with_ops=True)
      clipped_gradients, gradients_norm = tf.clip_by_global_norm(gradients, 5.0)
      self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
      self.train_summary = \
        tf.summary.merge(
          [tf.summary.scalar('train_loss', self.loss)])
    else:
      self.dropout_keep_prob = 1.0
      self.dropout_keep_prob = 1.0
      logits, sample_ids, _ = self.build_graph(mode)
      sample_words = reverse_target_table.lookup(tf.to_int64(sample_ids))
      self.predictions = tf.transpose(sample_words)
    self.saver = tf.train.Saver(tf.global_variables())

  def train(self, sess, batch):
    char_input, word_input, pos_input, target_in, target_out, word_len, char_len, target_len = zip(*batch)
    return sess.run(
      [self.update, self.loss, self.train_summary, self.global_step],
      {
        self.iterator.word_input: word_input,
        self.iterator.pos_input: pos_input,
        self.iterator.char_input: char_input,
        self.iterator.target_in: target_in,
        self.iterator.target_out: target_out,
        self.iterator.word_len: word_len,
        self.iterator.char_len: char_len,
        self.iterator.target_len: target_len,
      })
    pass

  def eval(self, sess, batch):
    char_input, word_input, pos_input, target_in, target_out, word_len, char_len, target_len = zip(*batch)
    return sess.run(
      [self.predictions],
      {
        self.iterator.word_input: word_input,
        self.iterator.pos_input: pos_input,
        self.iterator.char_input: char_input,
        self.iterator.word_len: word_len,
        self.iterator.char_len: char_len,
      })
    pass

  def build_graph(self, mode):
    encoded_chars = self.char_encoder()
    encoder_output, encoder_state = self.word_encoder(encoded_chars)
    return self.decoder(encoder_output, encoder_state, mode)

  def char_encoder(self):
    with tf.variable_scope("char_encoder"):
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
        [-1, self.hparams.max_seq_len, self.hparams.num_hidden_units * 2])
      return encoder_states

  def word_encoder(self, encoded_chars):
    words = tf.transpose(self.iterator.word_input)
    pos = tf.transpose(self.iterator.pos_input)
    encoded_chars = tf.transpose(encoded_chars, [1, 0, 2])
    # print(encoded_chars.get_shape().as_list())
    # TODO: add words vectors
    # word_vectors = tf.transpose(self.iterator.word_vectors, [1, 0, 2])
    pos_embedding = tf.get_variable(
      'pos_encoder_embedding', [self.vocabs.pos_vocab_size, self.hparams.word_embedding_size], tf.float32)
    pos_inputs = tf.nn.embedding_lookup(pos_embedding, pos)
    word_embedding = tf.get_variable(
      'word_encoder_embedding', [self.vocabs.word_vocab_size, self.hparams.word_embedding_size], tf.float32)
    word_inputs = tf.nn.embedding_lookup(word_embedding, words)
    inputs = tf.concat([word_inputs, encoded_chars, pos_inputs], -1)
    # inputs = word_vectors
    # self.embedding = word_inputs
    fwcell = self.create_cell(self.dropout_keep_prob)
    bwcell = self.create_cell(self.dropout_keep_prob)
    encoder_outputs, encoder_states = tf.nn.bidirectional_dynamic_rnn(
      fwcell, bwcell, inputs, dtype=tf.float32, sequence_length=self.iterator.word_len, time_major=True)
    return tf.concat(encoder_outputs, -1), encoder_states

  def decoder(self, encoder_outputs, encoder_state, mode):
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      target_input_len = tf.fill([tf.shape(self.iterator.target_len)[0]], self.hparams.max_seq_len + 2)
      target_in = tf.transpose(self.iterator.target_in)
      decoder_inputs = tf.reshape(
        tf.to_float(target_in),
        [tf.shape(target_in)[0], tf.shape(target_in)[1], 1],
        name='target_batch_input')
      helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, target_input_len, time_major=True)
    else:
      target_sos_id = tf.cast(self.vocabs.target_vocab['<s>'], tf.int32)
      target_eos_id = tf.cast(self.vocabs.target_vocab['</s>'], tf.int32)
      start_tokens = tf.fill([tf.size(self.iterator.word_len)], target_sos_id)
      end_token = target_eos_id
      helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        lambda ids: tf.reshape(tf.to_float(ids), [tf.shape(ids)[0], 1]),
        start_tokens,
        end_token)
    cell = self.create_cell(self.dropout_keep_prob, 2)
    decoder_init_state, cell = self.use_attention(cell, encoder_outputs, encoder_state)
    output_layer = layers_core.Dense(self.vocabs.target_vocab_size, use_bias=True, name='output_projection')
    decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_init_state, output_layer=output_layer)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        output_time_major=True,
        swap_memory=True,
      )
      # attention weight
      # self.logging = final_context_state.alignment_history.stack()
      return outputs.rnn_output, final_context_state
    else:
      outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        maximum_iterations=self.hparams.max_seq_len + 2,
        output_time_major=True,
        swap_memory=True,
      )
      return outputs.rnn_output, outputs.sample_id, final_context_state

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

  def create_attention(self, encoder_outputs):
    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
    num_units = self.hparams.num_hidden_units
    if self.hparams.attention_option == "luong":
      return tf.contrib.seq2seq.LuongAttention(
        num_units, attention_states, memory_sequence_length=self.iterator.word_len
      )
    elif self.hparams.attention_option == "bahdanau":
      return tf.contrib.seq2seq.BahdanauAttention(
        num_units, attention_states, memory_sequence_length=self.iterator.word_len
      )
    else:
      return None

  def use_attention(self, cell, encoder_outputs, encoder_state):
    attention = self.create_attention(encoder_outputs)
    if attention:
      cell = tf.contrib.seq2seq.AttentionWrapper(
        cell, attention, self.hparams.num_hidden_units, alignment_history=True)
      batch_size = tf.shape(tf.convert_to_tensor(encoder_state))[2]
      decoder_init_state = cell.zero_state(
        batch_size, tf.float32).clone(cell_state=encoder_state)
      return decoder_init_state, cell
    else: return encoder_state, cell

