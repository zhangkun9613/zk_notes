#coding=utf-8
import tensorflow as tf
import numpy as np
import collections
from src.utils import pinyin_converter

eos = '</s>'
sos = '<s>'
unk = "<unk>"

word_vocab_path = "vocabs/word.vocab"
pos_vocab_path = "vocabs/pos.vocab"
char_vocab_path = "vocabs/char.vocab"
target_vocab_path = "vocabs/target.vocab"

class Iterator(
  collections.namedtuple("iterator",
    ("word_input", "word_len", "char_input", "char_len", "pos_input", "target_input", "batch_size"))):
  pass

def initialize_vocabulary(vocab_path):
  if not tf.gfile.Exists(vocab_path):
    raise ValueError("Vocabulary file %s not found", vocab_path)
  with tf.gfile.GFile(vocab_path, mode="r") as f:
    rev_word_vocab = f.readlines()
    rev_word_vocab = [line.strip() for line in rev_word_vocab]
    word_vocab = dict([(y, x) for (x, y) in enumerate(rev_word_vocab)])
  return word_vocab, rev_word_vocab

def load_data(word_embedding_path, input_dir, max_char_len, max_seq_len, domain):
  # TODO: fasttext one word vector
  # word_embedding_model = FastText(word_embedding_path)
  word_vocab, rev_word_vocab = \
    initialize_vocabulary(input_dir + word_vocab_path)
  pos_vocab, rev_pos_vocab = \
    initialize_vocabulary(input_dir + pos_vocab_path)
  char_vocab, rev_char_vocab = \
    initialize_vocabulary(input_dir + char_vocab_path)
  target_vocab, rev_target_vocab = \
    initialize_vocabulary(input_dir + target_vocab_path)
  with tf.gfile.GFile(input_dir + domain + ".in", mode="r") as f:
    d = f.read().splitlines()
  word, pos, target = d[::3], d[1::3], d[2::3]
  word = [l.split(' ') for l in word]
  pos = [l.split(' ') for l in pos]
  word_len = [len(w) for w in word]
  pinyin = [pinyin_converter.pinyin_list_convert(l) for l in word]
  pinyin_len = [[len(w) for w in l]for l in pinyin]
  pinyin_len = [l + [0] * (max_seq_len - len(l))for l in pinyin_len]
  pinyin = [[list(w) + [eos] * (max_char_len - len(w)) for w in l] for l in pinyin]
  pinyin = [l + [[eos] * max_char_len] * (max_seq_len - len(l)) for l in pinyin]
  pinyin = [[[char_vocab[c] if c in char_vocab else char_vocab["<unk>"] for c in w] for w in l] for l in pinyin]
  word = [l + [eos] * (max_seq_len - len(l)) for l in word]
  word = [[word_vocab[w] if w in word_vocab else word_vocab["<unk>"] for w in l]for l in word]
  pos = [l + [eos] * (max_seq_len - len(l)) for l in pos]
  pos = [[pos_vocab[w] if w in pos_vocab else pos_vocab["<unk>"] for w in l]for l in pos]
  target = [target_vocab[w] if w in target_vocab else target_vocab["<unk>"]for w in target]
  pinyin = [l[:max_seq_len] for l in pinyin]
  word = [l[:max_seq_len] for l in word]
  pos = [l[:max_seq_len] for l in pos]
  word_len = [l if l <= max_seq_len else max_seq_len for l in word_len]
  pinyin_len = [l[:max_seq_len] for l in pinyin_len]
  return list(zip(pinyin, word, pos, word_len, pinyin_len, target))

def get_batch(data, batch_size, num_epochs, shuffle=True):
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

def get_serving_iterator(
    char_len_placeholder, char_placeholder, char_vocab_table, pos_placeholder,
    pos_vocab_table, word_placeholder, word_vocab_table, max_seq_len):
  char_input = char_placeholder[:max_seq_len]
  pos_input = pos_placeholder[:max_seq_len]
  word_input = word_placeholder[:max_seq_len]
  word_len = tf.size(word_placeholder)
  char_input = tf.cast(char_vocab_table.lookup(char_input), tf.int32)
  pos_input = tf.cast(pos_vocab_table.lookup(pos_input), tf.int32)
  word_input = tf.cast(word_vocab_table.lookup(word_input), tf.int32)
  # print(char_input.get_shape())
  return Iterator(
    char_input=tf.expand_dims(char_input, 0),
    word_input=tf.expand_dims(word_input, 0),
    pos_input=tf.expand_dims(pos_input, 0),
    target_input=None,
    word_len=tf.expand_dims(word_len, 0),
    char_len=tf.expand_dims(char_len_placeholder, 0),
    batch_size=1)

if __name__ == "__main__":
  x = "ab cd"
  y = tf.string_split([x], delimiter=' ').values
  z = tf.string_split(y, delimiter='').values
  with tf.Session() as sess:
    print(sess.run(y))
    print(y.get_shape())
