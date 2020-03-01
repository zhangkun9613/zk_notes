import tensorflow as tf
import fasttext
import numpy as np
import collections

from ..utils import data_utils

sos = '<s>'
eos = '</s>'
unk = '<unk>'

class Iterator(
  collections.namedtuple("Iterator", ("word_input", "pos_input", "wv_input", "dst_input",
                                      "dst_output", "src_seq_len", "dst_seq_len", "batch_size"))):
  pass

def load_data(fasttext_model_path, input_dir, domain, prefix, max_seq_len):
  fasttext_model = fasttext.load_model(fasttext_model_path)
  word_vocab, word_rev_vocab = data_utils.initialize_vocabulary(input_dir + 'word_' + domain + '.vocab')
  pos_vocab, pos_rev_vocab = data_utils.initialize_vocabulary(input_dir + 'pos_' + domain + '.vocab')
  dst_vocab, dst_rev_vocab = data_utils.initialize_vocabulary(input_dir + 'tag_' + domain + '.vocab')
  with tf.gfile.GFile(input_dir + prefix + '_' + domain + '.src') as f:
    d = f.read().splitlines()
  d = [l.strip().split("\t") for l in d]
  words = [l[0].split(" ")[:max_seq_len] for l in d]
  pos = [l[1].split(" ")[:max_seq_len] for l in d]
  if domain == 'specific_question':
    tags = [l[2].split(" ")[:max_seq_len] for l in d]
  else:
    tags = [l[2].split(" ")[:max_seq_len+1] for l in d]
  src_lens = [len(l) for l in words]
  dst_in = [[sos] + l for l in tags]
  dst_out = [l + [eos] for l in tags]
  dst_in_lens = [len(l) for l in dst_in]
  words = [l + [eos] * (max_seq_len-len(l)) for l in words]
  pos = [l + [eos] * (max_seq_len-len(l)) for l in pos]
  dst_in = [l + [eos] * (max_seq_len+2-len(l)) for l in dst_in]
  dst_out = [l + [eos] * (max_seq_len+2-len(l)) for l in dst_out]
  word_vectors = [[fasttext_model[w] for w in l] for l in words]
  words = [[word_vocab[w] if w in word_vocab else word_vocab[unk] for w in l] for l in words]
  pos = [[pos_vocab[w] if w in pos_vocab else pos_vocab[unk] for w in l] for l in pos]
  dst_in = [[dst_vocab[w] for w in l] for l in dst_in]
  dst_out = [[dst_vocab[w] for w in l] for l in dst_out]
  data = list(zip(words, word_vectors, pos, dst_in, dst_out, src_lens, dst_in_lens))
  return data, word_vocab, pos_vocab, dst_vocab


def batch_iterator(data, batch_size, num_epochs, shuffle=True):
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
  for epoch in range(num_epochs):
    if shuffle:
      shuffle_indices = np.random.permutation(np.arange(data_size))
      shuffle_data = data[shuffle_indices]
    else:
      shuffle_data = data
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      yield shuffle_data[start_index:end_index]


def get_serving_iterator(word_placeholder, wv_placeholder, pos_placeholder,
                         word_vocab_table, pos_vocab_table, max_seq_len):
  word_input = tf.string_split([word_placeholder]).values
  pos_input = tf.string_split([pos_placeholder]).values
  word_input = word_input[:max_seq_len]
  pos_input = pos_input[:max_seq_len]
  wv_input = wv_placeholder[:max_seq_len]
  word_input = tf.cast(word_vocab_table.lookup(word_input), tf.int32)
  pos_input = tf.cast(pos_vocab_table.lookup(pos_input), tf.int32)
  src_seq_len = tf.size(word_input)
  return Iterator(
    word_input=tf.expand_dims(word_input, 0),
    pos_input=tf.expand_dims(pos_input, 0),
    wv_input=tf.expand_dims(wv_input, 0),
    dst_input=None,
    dst_output=None,
    src_seq_len=tf.expand_dims(src_seq_len, 0),
    dst_seq_len=None,
    batch_size=1)
