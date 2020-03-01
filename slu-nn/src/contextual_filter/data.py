import tensorflow as tf
import os

prefix_vocab = ['<unk>', '<s>', '</s>']


def get_data(path):
  ret = []
  with tf.gfile.GFile(path) as f:
    d = f.read().splitlines()
    for i in range(0, len(d), 7):
      ret.append((d[i], d[i+1], d[i+2], d[i+3], d[i+4], d[i+5], d[i+6]))
  return ret


def create_vocab(out_path, data, name):
  if not tf.gfile.Exists(out_path):
    os.makedirs(out_path)
  assert len(name) == len(data[0]) / 2 + 1
  vocab_list = [{}, {}, {}, {}]

  for i in range(len(data[0])):
    d = [x[i] for x in data]
    if i == len(data[0]) - 1: vocab = vocab_list[3]
    else: vocab = vocab_list[i % 3]
    for line in d:
      words = line.split()
      for w in words:
        if w in vocab:
          vocab[w] += 1
        else:
          vocab[w] = 1

  for i in range(len(name)):
    with tf.gfile.GFile(out_path + name[i] + ".vocab", "w+") as f:
      v = prefix_vocab + sorted(vocab_list[i], key=vocab_list[i].get, reverse=True)
      f.write('\n'.join(v))


def write_data(out_path, data, name):
  if not tf.gfile.Exists(out_path):
    os.makedirs(out_path)
  with tf.gfile.GFile(out_path + name + '.src', "w+") as f:
    for i in range(len(data)):
      f.write('\t'.join(data[i]))
      f.write('\n')

if __name__ == "__main__":
  data_dir = "../data/"
  train_file = "train.txt"
  dev_file = "dev.txt"
  test_file = "test.txt"
  vocab_out_path = "../data/vocabs/"
  out_path = "../data/"
  train_data = get_data(data_dir + train_file)
  dev_data = get_data(data_dir + dev_file)
  test_data = get_data(data_dir + test_file)
  data = train_data
  data.extend(dev_data)
  data.extend(test_data)
  create_vocab(vocab_out_path, data, ['word', 'pos', 'tag', 'help'])
  write_data(out_path, train_data, 'train')
  write_data(out_path, dev_data, 'dev')
  write_data(out_path, test_data, 'test')
