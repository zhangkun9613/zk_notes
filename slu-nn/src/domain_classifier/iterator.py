import tensorflow as tf

import collections

sos = '<s>'
eos = '</s>'

class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "word_input",
                                           "pos_input",
                                           "dst_output",
                                           "src_seq_len"
                                           ))):
  pass


class ServingInput(collections.namedtuple("ServingInput",
                                          ("word_input",
                                           "pos_input",
                                           "src_seq_len"))):
  pass


def get_iterator(dataset,
                 word_vocab_table,
                 pos_vocab_table,
                 dst_vocab_table,
                 dst_vocab_size,
                 batch_size,
                 max_seq_len=20,
                 random_seed=42):
  output_buffer_size = batch_size * 1000
  word_eos_id = tf.cast(
    word_vocab_table.lookup(tf.constant(eos)),
    tf.int32
  )
  pos_eos_id = tf.cast(
    pos_vocab_table.lookup(tf.constant(eos)),
    tf.int32
  )
  dataset = dataset.shuffle(output_buffer_size, random_seed)
  dataset = dataset.map(
    lambda src: tf.split(tf.string_split([src], "\t").values, 3, 0),
    output_buffer_size=output_buffer_size
  )
  dataset = dataset.map(
    lambda src, pos, dst: (src[0], pos[0], dst[0]),
    output_buffer_size=output_buffer_size
  )
  dataset = dataset.map(
    lambda src, pos, dst: (tf.string_split([src]).values, tf.string_split([pos]).values, dst),
    output_buffer_size=output_buffer_size
  )
  dataset = dataset.map(
    lambda src, pos, dst: (src, pos, dst, tf.size(src)),
    output_buffer_size=output_buffer_size
  )
  dataset = dataset.map(
    lambda src, pos, dst, src_seq: (tf.concat([src[:max_seq_len], [eos]*max_seq_len], 0)[:max_seq_len],
                                    tf.concat([pos[:max_seq_len], [eos]*max_seq_len], 0)[:max_seq_len],
                                    dst,
                                    src_seq),
    output_buffer_size=output_buffer_size
  )
  dataset = dataset.map(
    lambda src, pos, dst, src_seq: (tf.cast(word_vocab_table.lookup(src), tf.int32),
                                    tf.cast(pos_vocab_table.lookup(pos), tf.int32),
                                    tf.cast(dst_vocab_table.lookup(dst), tf.int32),
                                    src_seq),
    output_buffer_size=output_buffer_size
  )
  dataset = dataset.map(
    lambda src, pos, dst, src_seq: (src, pos, tf.one_hot(dst, dst_vocab_size), src_seq),
    output_buffer_size=output_buffer_size
  )
  dataset = dataset.repeat()
  batched_dataset = dataset.padded_batch(
    batch_size,
    padded_shapes=(tf.TensorShape([None]),
                   tf.TensorShape([None]),
                   tf.TensorShape([None]),
                   tf.TensorShape([])),
    padding_values=(word_eos_id, pos_eos_id, 0.0, 0)
  )
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, pos_ids, dst_ids, src_seq_len) = (batched_iter.get_next())
  return BatchedInput(
    initializer = batched_iter.initializer,
    word_input = src_ids,
    pos_input = pos_ids,
    dst_output = dst_ids,
    src_seq_len = src_seq_len
  )


def get_infer_iterator(word_dataset,
                       pos_dataset,
                       word_vocab_table,
                       pos_vocab_table,
                       batch_size,
                       max_seq_len=20):
  word_eos_id = tf.cast(
    word_vocab_table.lookup(tf.constant(eos)),
    tf.int32
  )
  pos_eos_id = tf.cast(
    pos_vocab_table.lookup(tf.constant(eos)),
    tf.int32
  )
  dataset = tf.contrib.data.Dataset.zip((word_dataset, pos_dataset))
  dataset = dataset.map(lambda src, pos: (tf.string_split([src]).values,
                                          tf.string_split([pos]).values))
  dataset = dataset.map(lambda src, pos: (src, pos, tf.size(src)))
  dataset = dataset.map(lambda src, pos, src_seq: (
    tf.concat([src[:max_seq_len], [eos] * max_seq_len], 0)[:max_seq_len],
    tf.concat([pos[:max_seq_len], [eos] * max_seq_len], 0)[:max_seq_len],
    src_seq))
  dataset = dataset.map(lambda src, pos, src_seq: (tf.cast(word_vocab_table.lookup(src), tf.int32),
                                                   tf.cast(pos_vocab_table.lookup(pos), tf.int32),
                                                   src_seq))
  batched_dataset = dataset.padded_batch(
    batch_size,
    padded_shapes=(tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])),
    padding_values=(word_eos_id, pos_eos_id, 0)
  )
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, pos_ids, src_seq_len) = batched_iter.get_next()
  return BatchedInput(
    initializer = batched_iter.initializer,
    word_input = src_ids,
    pos_input = pos_ids,
    dst_output = None,
    src_seq_len = src_seq_len
  )

def get_serving_iterator(word_input, pos_input, word_vocab_table, pos_vocab_table, max_seq_len=20):
  word_input = tf.string_split([word_input]).values
  pos_input = tf.string_split([pos_input]).values
  word_input = word_input[:max_seq_len]
  src_seq_len = tf.size(word_input)
  word_input = tf.concat([word_input, [eos] * max_seq_len], 0)[:max_seq_len]
  pos_input = tf.concat([pos_input, [eos] * max_seq_len], 0)[:max_seq_len]
  word_input = tf.cast(word_vocab_table.lookup(word_input), tf.int32)
  pos_input = tf.cast(pos_vocab_table.lookup(pos_input), tf.int32)
  return ServingInput(
    word_input = tf.expand_dims(word_input, 0),
    pos_input = tf.expand_dims(pos_input, 0),
    src_seq_len = tf.expand_dims(src_seq_len, 0)
  )
