import tensorflow as tf
import numpy as np
import collections
import fasttext

sos = '<s>'
eos = '</s>'
unk = '<unk>'

class History(
  collections.namedtuple('History', ('word_input', 'word_vector', 'pos_input', 'tag_input',
                                     'word_len', 'help', 'help_len', 'batch_size'))):
  pass

class Current(
  collections.namedtuple('Current', ('word_input', 'word_vector', 'pos_input',
                                     'tag_input', 'word_len', 'batch_size'))):
  pass

def initialize_vocab(vocab_path):
  if tf.gfile.Exists(vocab_path):
    with tf.gfile.GFile(vocab_path) as f:
      rev_vocab = f.read().splitlines()
    # vocabs = dict([(y, x) for (x, y) in enumerate(rev_vocab)])
    vocab = {y: x for (x, y) in enumerate(rev_vocab)}
  else:
    raise ValueError("Vocab file %s not found, Please run data.py first" % (vocab_path,))
  # print (rev_vocab)
  # print (vocab)
  return vocab, rev_vocab


def load_data(embedding_model, input_dir, max_seq_len, data_file='train.src'):
  embedding_model = fasttext.load_model(embedding_model)
  word_vocab, word_rev_vocab = initialize_vocab(input_dir + "vocabs/word.vocab")
  pos_vocab, pos_rev_vocab = initialize_vocab(input_dir + "vocabs/pos.vocab")
  tag_vocab, tag_rev_vocab = initialize_vocab(input_dir + "vocabs/tag.vocab")
  help_vocab, help_rev_vocab = initialize_vocab(input_dir + "vocabs/help.vocab")
  with tf.gfile.GFile(input_dir + data_file) as f:
    d = f.read().splitlines()
  d = [l.split('\t') for l in d]
  pre_words = [l[0].split(' ')[:max_seq_len] for l in d]
  pre_pos = [l[1].split(' ')[:max_seq_len] for l in d]
  pre_tags = [l[2].split(' ')[:max_seq_len] for l in d]
  words = [l[3].split(' ')[:max_seq_len] for l in d]
  pos = [l[4].split(' ')[:max_seq_len] for l in d]
  tags = [l[5].split(' ')[:max_seq_len] for l in d]
  help = [l[6].split(' ')[:max_seq_len] for l in d]
  help_len = [len(l) for l in help]
  pre_words_len = [len(l) for l in pre_words]
  words_len = [len(l) for l in words]
  words = [l + [eos] * (max_seq_len - len(l)) for l in words]
  pos = [l + [eos] * (max_seq_len - len(l)) for l in pos]
  tags = [l + [eos] * (max_seq_len - len(l)) for l in tags]
  help = [l + [eos] * (max_seq_len - len(l)) for l in help]
  pre_words = [l + [eos] * (max_seq_len - len(l)) for l in pre_words]
  pre_pos = [l + [eos] * (max_seq_len - len(l)) for l in pre_pos]
  pre_tags = [l + [eos] * (max_seq_len - len(l)) for l in pre_tags]
  pre_word_vectors = [[embedding_model[w] for w in l] for l in pre_words]
  word_vectors = [[embedding_model[w] for w in l] for l in words]
  pre_words = [[word_vocab[w] for w in l] for l in pre_words]
  words = [[word_vocab[w] for w in l] for l in words]
  pre_pos = [[pos_vocab[w] for w in l] for l in pre_pos]
  pos = [[pos_vocab[w] for w in l] for l in pos]
  pre_tags = [[tag_vocab[w] for w in l] for l in pre_tags]
  tags = [[tag_vocab[w] for w in l] for l in tags]
  help = [[help_vocab[w] for w in l] for l in help]
  pre_data = list(zip(pre_words, pre_word_vectors, pre_pos, pre_tags, pre_words_len, help, help_len))
  cur_data = list(zip(words, word_vectors, pos, tags, words_len))
  # print (pre_data)
  # print (cur_data)
  return pre_data, cur_data, word_vocab, pos_vocab, tag_vocab, help_vocab

def get_batch(cur_data, pre_data, batch_size, num_epochs, shuffle=True):
  pre_data = np.array(pre_data)
  cur_data = np.array(cur_data)
  data_size = len(pre_data)
  num_batches_per_epoch = int((data_size - 1) / batch_size + 1)
  for _ in range(num_epochs):
    if shuffle:
      shuffle_indices = np.random.permutation(np.arange(data_size))
      shuffle_cur_data = cur_data[shuffle_indices]
      shuffle_pre_data = pre_data[shuffle_indices]
    else:
      shuffle_pre_data = pre_data
      shuffle_cur_data = cur_data
    for batch_num in range(num_batches_per_epoch):
      start = batch_size * batch_num
      end = min((batch_num + 1) * batch_size, data_size)
      yield shuffle_pre_data[start:end], shuffle_cur_data[start:end]

def get_serving_input(c_word_placeholder, c_pos_placeholder, c_tag_placeholder, c_wv_placehodler,
                      h_word_placeholder, h_pos_placeholder, h_tag_placeholder, h_wv_placehodler,
                      word_vocab_table, pos_vocab_table, tag_vocab_table, max_seq_len, embedding_model):
  embedding_model = fasttext.load_model(embedding_model)
  eos_wv = embedding_model['</s>']
  c_word_input = tf.string_split([c_word_placeholder]).values
  c_src_seq_len = tf.size(c_word_input)
  c_word_input = tf.concat([c_word_input[:max_seq_len], [eos] * max_seq_len], 0)[:max_seq_len]
  c_pos_input = tf.string_split([c_pos_placeholder]).values
  c_pos_input = tf.concat([c_pos_input[:max_seq_len], [eos] * max_seq_len], 0)[:max_seq_len]
  c_tag_input = tf.string_split([c_tag_placeholder]).values
  c_tag_input = tf.concat([c_tag_input[:max_seq_len], [eos] * max_seq_len], 0)[:max_seq_len]
  c_wv_input = tf.concat([c_wv_placehodler[:max_seq_len], [eos_wv] * max_seq_len], 0)[:max_seq_len]
  c_word_input = tf.cast(word_vocab_table.lookup(c_word_input), tf.int32)
  c_pos_input = tf.cast(pos_vocab_table.lookup(c_pos_input), tf.int32)
  c_tag_input = tf.cast(tag_vocab_table.lookup(c_tag_input), tf.int32)
  h_word_input = tf.string_split([h_word_placeholder]).values
  h_src_seq_len = tf.size(h_word_input)
  h_word_input = tf.concat([h_word_input[:max_seq_len], [eos] * max_seq_len], 0)[:max_seq_len]
  h_pos_input = tf.string_split([h_pos_placeholder]).values
  h_pos_input = tf.concat([h_pos_input[:max_seq_len], [eos] * max_seq_len], 0)[:max_seq_len]
  h_tag_input = tf.string_split([h_tag_placeholder]).values
  h_tag_input = tf.concat([h_tag_input[:max_seq_len], [eos] * max_seq_len], 0)[:max_seq_len]
  h_wv_input = tf.concat([h_wv_placehodler[:max_seq_len], [eos_wv] * max_seq_len], 0)[:max_seq_len]
  h_word_input = tf.cast(word_vocab_table.lookup(h_word_input), tf.int32)
  h_pos_input = tf.cast(pos_vocab_table.lookup(h_pos_input), tf.int32)
  h_tag_input = tf.cast(tag_vocab_table.lookup(h_tag_input), tf.int32)
  current = Current(
    word_input=tf.expand_dims(c_word_input, 0),
    pos_input=tf.expand_dims(c_pos_input, 0),
    word_vector=tf.expand_dims(c_wv_input, 0),
    tag_input=tf.expand_dims(c_tag_input, 0),
    word_len=tf.expand_dims(c_src_seq_len, 0),
    batch_size=1)
  history = History(
    word_input=tf.expand_dims(h_word_input, 0),
    pos_input=tf.expand_dims(h_pos_input, 0),
    word_vector=tf.expand_dims(h_wv_input, 0),
    tag_input=tf.expand_dims(h_tag_input, 0),
    word_len=tf.expand_dims(h_src_seq_len, 0),
    help=None,
    help_len=None,
    batch_size=1)
  return current, history

