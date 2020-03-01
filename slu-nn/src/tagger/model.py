import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops import lookup_ops
from tensorflow.python.layers import core as layers_core

class Model(object):
  def __init__(self, hparams, vocabs, iterator, mode, reverse_dst_table):
    self.hparams = hparams
    self.vocabs = vocabs
    self.iterator = iterator
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      self.dropout_keep_prob = hparams.dropout_keep_prob
    else:
      self.dropout_keep_prob = 1.0
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      logits, _ = self.build_graph(mode)
      labels = tf.transpose(iterator.dst_output)
      crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
      output_weights = tf.sequence_mask(iterator.dst_seq_len, hparams.max_seq_len + 2, dtype=logits.dtype)
      output_weights = tf.transpose(output_weights)
      self.loss = tf.reduce_sum(crossent * output_weights) / tf.to_float(iterator.batch_size)
    else:
      logits, sample_ids, _ = self.build_graph(mode)
      sample_words = reverse_dst_table.lookup(tf.to_int64(sample_ids))
      self.predictions = tf.transpose(sample_words)

    self.global_step = tf.Variable(0, trainable=False)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      params = tf.trainable_variables()
      opt = tf.train.AdamOptimizer(hparams.learning_rate)
      gradients = tf.gradients(self.loss, params, colocate_gradients_with_ops=True)
      clipped_gradients, gradients_norm = tf.clip_by_global_norm(gradients, 5.0)
      self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
      self.train_summary = tf.summary.merge([tf.summary.scalar('train_loss', self.loss)])

    self.saver = tf.train.Saver(tf.global_variables())

  def train(self, sess, batch):
    word_input, wv_input, pos_input, dst_input, dst_output, src_seq_len, dst_seq_len = zip(*batch)
    return sess.run([self.update, self.loss, self.train_summary, self.global_step],
                    {self.iterator.word_input: word_input,
                     self.iterator.wv_input: wv_input,
                     self.iterator.pos_input: pos_input,
                     self.iterator.dst_input: dst_input,
                     self.iterator.dst_output: dst_output,
                     self.iterator.src_seq_len: src_seq_len,
                     self.iterator.dst_seq_len: dst_seq_len
                    })

  def eval(self, sess, batch):
    word_input, wv_input, pos_input, _, _, src_seq_len, _ = zip(*batch)
    return sess.run([self.predictions],
                    {self.iterator.word_input: word_input,
                     self.iterator.wv_input: wv_input,
                     self.iterator.pos_input: pos_input,
                     self.iterator.src_seq_len: src_seq_len
                    })

  def create_cell(self, dropout_keep, layers=1):
    cells = []
    for i in range(layers):
      cell = rnn.LSTMCell(self.hparams.rnn_size, forget_bias=1.0)
      if dropout_keep < 1.0:
        cell = rnn.DropoutWrapper(cell, dropout_keep, dropout_keep)
      cells.append(cell)
    if len(cells) == 1:
      return cells[0]
    else:
      return rnn.MultiRNNCell(cells, state_is_tuple=True)


  def build_graph(self, mode):
    encoder_outputs, encoder_state = self.create_encoder()
    return self.create_decoder(encoder_outputs, encoder_state, mode)


  def create_encoder(self):
    word_input = tf.transpose(self.iterator.word_input)
    pos_input = tf.transpose(self.iterator.pos_input)
    wv_input = tf.transpose(self.iterator.wv_input, [1, 0, 2])
    words_embedding = tf.get_variable('encoder_embedding', [self.vocabs.word_vocab_size, self.hparams.word_embedding_size], tf.float32)
    inputs = tf.nn.embedding_lookup(words_embedding, word_input)
    pos_embedding = tf.get_variable('encoder_pos_embedding', [self.vocabs.pos_vocab_size, self.hparams.pos_embedding_size], tf.float32)
    pos_inputs = tf.nn.embedding_lookup(pos_embedding, pos_input)
    inputs = tf.concat([inputs, wv_input, pos_inputs], -1)
    fwcell = self.create_cell(self.dropout_keep_prob)
    bwcell = self.create_cell(self.dropout_keep_prob)
    encoder_outputs, encoder_states = tf.nn.bidirectional_dynamic_rnn(
      fwcell, bwcell, inputs, dtype=tf.float32, sequence_length=self.iterator.src_seq_len, time_major=True)
    return tf.concat(encoder_outputs, -1), encoder_states
    # return tf.concat(encoder_outputs, -1), tf.concat(encoder_states, -1),

  def create_decoder(self, encoder_outputs, encoder_state, mode):
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      dst_input_len = tf.fill([tf.shape(self.iterator.dst_seq_len)[0]], self.hparams.max_seq_len + 2)
      dst_input = tf.transpose(self.iterator.dst_input)
      decoder_inputs = tf.reshape(
        tf.to_float(dst_input),
        [tf.shape(dst_input)[0], tf.shape(dst_input)[1], 1],
        name='dst_batch_input')
      helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, dst_input_len, time_major=True)
    else:
      dst_sos_id = tf.cast(self.vocabs.dst_vocab['<s>'], tf.int32)
      dst_eos_id = tf.cast(self.vocabs.dst_vocab['</s>'], tf.int32)
      # print(tf.shape(self.iterator.src_seq_len))
      start_tokens = tf.fill([tf.shape(self.iterator.src_seq_len)[0]], dst_sos_id)
      end_token = dst_eos_id
      helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        lambda ids: tf.reshape(tf.to_float(ids), [tf.shape(ids)[0], 1]),
        start_tokens,
        end_token
      )
    cell = self.create_cell(self.dropout_keep_prob, 2)
    decoder_init_state, cell = self.use_attention(cell, encoder_outputs, encoder_state)
    output_layer = layers_core.Dense(self.vocabs.dst_vocab_size, use_bias=True, name='output_projection')
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

  def create_attention(self, encoder_outputs):
    attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
    num_units = self.hparams.rnn_size
    if self.hparams.attention_option == "luong":
      return tf.contrib.seq2seq.LuongAttention(
        num_units, attention_states, memory_sequence_length=self.iterator.src_seq_len
      )
    elif self.hparams.attention_option == "bahdanau":
      return tf.contrib.seq2seq.BahdanauAttention(
        num_units, attention_states, memory_sequence_length=self.iterator.src_seq_len
      )
    else:
      return None

  def use_attention(self, cell, encoder_outputs, encoder_state):
    attention = self.create_attention(encoder_outputs)
    if attention:
      cell = tf.contrib.seq2seq.AttentionWrapper(
        cell, attention, self.hparams.rnn_size, alignment_history=True)
      batch_size = tf.shape(tf.convert_to_tensor(encoder_state))[2]
      decoder_init_state = cell.zero_state(
        batch_size, tf.float32).clone(cell_state=encoder_state)
      return decoder_init_state, cell
    else: return encoder_state, cell
