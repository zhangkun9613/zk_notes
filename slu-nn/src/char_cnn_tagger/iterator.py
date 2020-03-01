#coding=utf-8
import collections
from pyfasttext import FastText
from src.utils import pinyin_converter
import numpy as np
import tensorflow as tf

sos = "<s>"
eos = "</s>"
unk = "<unk>"

word_vocab_path = "vocabs/word.vocab"
char_vocab_path = "vocabs/char.vocab"
target_vocab_path = "vocabs/target.vocab"
pos_vocab_path = "vocabs/pos.vocab"

class Iterator(collections.namedtuple("iterator", (
  "word_input", "pos_input", "word_len", "char_input", "char_len", "target_in", "target_out", "target_len", "batch_size"))):
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
  char_vocab, rev_char_vocab = \
    initialize_vocabulary(input_dir + char_vocab_path)
  target_vocab, rev_target_vocab = \
    initialize_vocabulary(input_dir + target_vocab_path)
  pos_vocab, rev_pos_vocab = \
    initialize_vocabulary(input_dir + pos_vocab_path)
  with tf.gfile.GFile(input_dir + domain + ".in", mode="r") as f:
    d = f.read().splitlines()
  word, pos, target = d[::3], d[1::3], d[2::3]
  word = [l.split(' ') for l in word]
  pos = [l.split(' ') for l in pos]
  target = [l.split(' ') for l in target]
  word_len = [len(w) for w in word]
  target_len = [len(w) + 1 for w in target]
  pinyin = [pinyin_converter.pinyin_list_convert(l) for l in word]
  pinyin_len = [[len(w) for w in l]for l in pinyin]
  pinyin_len = [l + [0] * (max_seq_len - len(l))for l in pinyin_len]
  pinyin = [[list(w) + [eos] * (max_char_len - len(w)) for w in l] for l in pinyin]
  pinyin = [l + [[eos] * max_char_len] * (max_seq_len - len(l)) for l in pinyin]
  pinyin = [[[char_vocab[c] if c in char_vocab else char_vocab["<unk>"] for c in w] for w in l] for l in pinyin]
  word = [l + [eos] * (max_seq_len - len(l)) for l in word]
  pos = [l + [eos] * (max_seq_len - len(l)) for l in pos]
  word = [[word_vocab[w] if w in word_vocab else word_vocab["<unk>"] for w in l]for l in word]
  pos = [[pos_vocab[w] if w in pos_vocab else pos_vocab["<unk>"] for w in l]for l in pos]
  target_in = [[sos] + l + [eos] * (max_seq_len + 1 - len(l)) for l in target]
  target_in = [[target_vocab[w] if w in target_vocab else target_vocab["<unk>"] for w in l]for l in target_in]
  target_out = [l + [eos] * (max_seq_len + 2 - len(l)) for l in target]
  target_out = [[target_vocab[w] if w in target_vocab else target_vocab["<unk>"] for w in l]for l in target_out]
  return list(zip(pinyin, word, pos, target_in, target_out, word_len, pinyin_len, target_len))

def get_batch(data, batch_size, num_epoch, shuffle=True):
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
  for _ in range(num_epoch):
    if shuffle:
      shuffle_indices = np.random.permutation(np.arange(data_size))
      shuffle_data = data[shuffle_indices]
    else:
      shuffle_data = data
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      yield shuffle_data[start_index: end_index]

if __name__ == "__main__":
  print([eos] * 2)
