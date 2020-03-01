import tensorflow as tf

class SiameseLSTM(object):

  def __init__(self, max_len, embedding_size, hidden_units, l1_reg_lambda, batch_size, training):
    self.input_x1_wv = tf.placeholder(tf.float32, [None, max_len, embedding_size], name="input_x1_wv")
    self.input_x2_wv = tf.placeholder(tf.float32, [None, max_len, embedding_size], name="input_x2_wv")
    self.input_x1_pv = tf.placeholder(tf.float32, [None, max_len, 46], name="input_x1_pv")
    self.input_x2_pv = tf.placeholder(tf.float32, [None, max_len, 46], name="input_x2_pv")
    input_x1 = tf.concat([self.input_x1_wv, self.input_x1_pv], -1)
    input_x2 = tf.concat([self.input_x2_wv, self.input_x2_pv], -1)
    self.x1_seq_len = tf.placeholder(tf.int32, [None], name="x1_seq_len")
    self.x2_seq_len = tf.placeholder(tf.int32, [None], name="x2_seq_len")
    self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
    if training:
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    else:
      self.dropout_keep_prob = tf.constant(1.0, tf.float32)

    #l1_loss = tf.constant(0.0, name="l1_loss")
    #l1_regularizer = tf.contrib.layers.l1_regularizer(scale=l1_reg_lambda)

    with tf.name_scope("output"):
      self.out1 = self.stackedRNN(input_x1, self.x1_seq_len, self.dropout_keep_prob, "side1",
                                  max_len, hidden_units)
      self.out2 = self.stackedRNN(input_x2, self.x2_seq_len, self.dropout_keep_prob, "side2",
                                  max_len, hidden_units)
      self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keep_dims=True))

     # self.distance = tf.div(self.distance,tf.sqrt(tf.reduce_sum(tf.add(tf.square(self.out1),tf.square(self.out2)), 1, keep_dims=True)))
      self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
                                                   tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))
      self.distance = tf.reshape(self.distance, [-1], name="distance")
    with tf.name_scope("loss"):
      #weights = tf.trainable_variables()
      #regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
      #self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size) + regularization_penalty
      self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size)

    with tf.name_scope("accuracy"):
      self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance), name="temp_sim")
      correct_predictions = tf.equal(self.temp_sim, self.input_y)
      self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

  def stackedRNN(self, x, x_seq_len, dropout, scope, sequence_length, hidden_units):
    n_hidden = hidden_units
    n_layers = 2
    with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
      stacked_rnn_fw = []
      for _ in range(n_layers):
        fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, use_peepholes=True, forget_bias=1.0, state_is_tuple=True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
        stacked_rnn_fw.append(lstm_fw_cell)
      lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
      outputs, _ = tf.nn.dynamic_rnn(lstm_fw_cell_m, x, sequence_length=x_seq_len, dtype=tf.float32)
    batch_size = tf.shape(outputs)[0]
    input_size = int(outputs.get_shape()[2])
    index = tf.range(0, batch_size) * sequence_length + (x_seq_len - 1)
    flat = tf.reshape(outputs, [-1, input_size])
    return tf.gather(flat, index)

  def contrastive_loss(self, y, d, batch_size):
    gamma = 1
    tmp = tf.pow(d,gamma) * y * tf.square(d)
    tmp2 = tf.pow(1-d,gamma) * (1 - y) * tf.square(tf.maximum((1 - d), 0))
    return tf.reduce_sum(tmp + tmp2) / batch_size / 2