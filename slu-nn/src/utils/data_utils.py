import tensorflow as tf
import random
import os

is_product = True

def initialize_vocabulary(vocabulary_path):
  if tf.gfile.Exists(vocabulary_path):
    rev_vocab = []
    with tf.gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def create_vocab(path, data, index, name, prefix_vocab=None, suffix_vocab=None):
  if suffix_vocab is None:
    suffix_vocab = []
  if prefix_vocab is None:
    prefix_vocab = ['<unk>', '<s>', '</s>']
  vocab = {}
  if data is not None:
    data = [x[index] for x in data]
    for d in data:
      words = d.split()
      for w in words:
        if w in vocab:
          vocab[w] += 1
        else:
          vocab[w] = 1
  with tf.gfile.GFile(path + name + '.vocab', "w+") as f:
    v = prefix_vocab + sorted(vocab, key=vocab.get, reverse=True)[: 7000] + suffix_vocab
    f.write("\n".join(v))


def prepare_domain_data(
    unknown_data_file,
    media_data_file,
    command_data_file,
    question_data_file,
    output_dir, random_seed):
  def get_data(path, domain_name):
    ret = []
    with tf.gfile.GFile(path, "r") as f:
      d = f.read().splitlines()
      for i in range(0, len(d), 2):
        ret.append((d[i], d[i+1], domain_name))
    random.shuffle(ret)
    return ret

  def write_data(path, data):
    with tf.gfile.GFile(path, "w+") as f:
      d = [x[0] + "\t" + x[1] + "\t" + x[2] for x in data]
      f.write("\n".join(d))

  random.seed(random_seed)
  unknown_data = get_data(unknown_data_file, "UNKNOWN")
  media_data = get_data(media_data_file, "MEDIA")
  command_data = get_data(command_data_file, "COMMAND")
  question_data = get_data(question_data_file, "QUESTION")
  data = unknown_data[:int(len(unknown_data) * 0.2)] + (media_data * 3) + (question_data * 1) + (command_data * 1)
  random.shuffle(data)

  dev_data = data[:int(len(data) * 0.1)]
  test_data = data[int(len(data) * 0.1):int(len(data) * 0.2)]
  if is_product:
    train_data = data[int(len(data) * 0):] + command_data * 3
  else:
    train_data = data[int(len(data) * 0):] + command_data
  random.shuffle(train_data)
  write_data(output_dir + "dev.src", dev_data)
  write_data(output_dir + "test.src", test_data)
  write_data(output_dir + "train.src", train_data)
  create_vocab(output_dir, train_data, 0, "word")
  create_vocab(output_dir, train_data, 1, "pos")
  create_vocab(output_dir, None, None, "label", ['UNKNOWN', 'MEDIA', 'COMMAND', 'QUESTION'])


def prepare_tagging_data(data_file, output_dir, domain_name, folds, random_seed):
  def get_data(path):
    ret = []
    with tf.gfile.GFile(path, "r") as f:
      d = f.read().splitlines()
      for i in range(0, len(d), 4):
        ret.append((d[i], d[i+1], d[i+2], d[i+3]))
    return ret

  def write_data(path, data, domain_name):
    with tf.gfile.GFile(path + "_" + domain_name + '.src', "w+") as f:
      d = [x[0] + "\t" + x[1] + "\t" + x[2] + " " + x[3] for x in data]
      f.write("\n".join(d))

  def do_create_vocab(path, data, domain_name):
    create_vocab(path, data, 0, 'word_' + domain_name)
    create_vocab(path, data, 1, 'pos_' + domain_name)
    create_vocab(path, data, 2, 'tag_' + domain_name)
    create_vocab(path, data, 3, 'label_' + domain_name)

  data = get_data(data_file)
  random.seed(random_seed)
  random.shuffle(data)
  dev_data = data[:int(len(data) * folds[0])]
  test_data = data[int(len(data) * folds[1][0]):int(len(data) * folds[1][1])]
  train_data = data[int(len(data) * folds[2]):]
  write_data(output_dir + 'dev', dev_data, domain_name)
  write_data(output_dir + 'test', test_data, domain_name)
  write_data(output_dir + 'train', train_data, domain_name)
  do_create_vocab(output_dir, train_data, domain_name)


def prepare_tagging_data_wo_label(data_file, output_dir, domain_name, folds, random_seed):
  data = []
  with tf.gfile.GFile(data_file, 'r') as f:
    d = f.read().splitlines()
    d = [i.split('\t') for i in d if len(i.strip()) > 0]
    for i in d:
      data.append((i[0], i[1], i[2]))
  random.seed(random_seed)
  random.shuffle(data)
  def write_data(path, data, domain_name):
    with tf.gfile.GFile(path + '_' + domain_name + '.src', 'w+') as f:
      d = [i[0] + '\t' + i[1] + '\t' + i[2] for i in data]
      f.write('\n'.join(d))
  dev_data = data[:int(len(data) * folds[0])]
  test_data = data[int(len(data) * folds[1][0]):int(len(data) * folds[1][1])]
  train_data = data[int(len(data) * folds[2]):]
  write_data(os.path.join(out_dir, 'dev'), dev_data, domain_name)
  write_data(os.path.join(out_dir, 'test'), test_data, domain_name)
  write_data(os.path.join(out_dir, 'train'), train_data, domain_name)

  create_vocab(output_dir, data, 0, 'word_' + domain_name)
  create_vocab(output_dir, data, 1, 'pos_' + domain_name)
  create_vocab(output_dir, data, 2, 'tag_' + domain_name)

def prepare_test_tagging_data_wo_label(data_file, output_dir, domain_name):
  data = []
  with tf.gfile.GFile(data_file, mode='r') as f:
    d = f.read().splitlines()
    d = [i.split('\t') for i in d if len(i.strip()) > 0]
    for i in d:
      data.append((i[0], i[1]))
  with tf.gfile.GFile(os.path.join(output_dir, 'test_plus_'+domain_name+'.src'), 'w+') as f:
    d = [i[0] + '\t' + i[1] for i in data]
    f.write('\n'.join(d))

def add_label():
    domains = ['question', 'media', 'command']
    for domain in domains:
        labels = []
        with open('../../training/tagging/label_' + domain + '.vocab', 'r') as f:
            lines = f.readlines()
            for i in range(3, len(lines)):
                labels.append(lines[i].strip())
        labels = '\n'.join(labels)
        with open('../../training/tagging/tag_' + domain + '.vocab', 'a') as f:
            f.write('\n' + labels)


if __name__ == "__main__":
  seed = 42

  out_dir = '../../training/tagging'
  if not os.path.isdir(out_dir):
      os.makedirs(out_dir)


  # out_dir = '../../training/tagging'
  # if not os.path.isdir(out_dir):
  #   os.makedirs(out_dir)


  domain_data_dir = "../../data/domain_classification/"
  prepare_domain_data(domain_data_dir + 'unknown.txt',
                      domain_data_dir + 'media.txt',
                      domain_data_dir + 'command.txt',
                      domain_data_dir + 'question.txt',
                      "../../training/domain/", seed)


  tagging_data_dir = "../../data/slot_filling/"
  tagging_test_data_file = "../../data/siamese_classification/train_plus.txt"


  prepare_tagging_data(tagging_data_dir + 'command.txt',
                       '../../training/tagging/', 'command',
                       (0.1, (0.1, 0.2), 0 if is_product else 0.2), seed)
  prepare_tagging_data(tagging_data_dir + 'media.txt',
                       '../../training/tagging/', 'media',
                      (0.1, (0.1, 0.2), 0 if is_product else 0.2), seed)
  prepare_tagging_data(tagging_data_dir + 'question.txt',
                       '../../training/tagging/', 'question',
                       (0.1, (0.1, 0.2), 0 if is_product else 0.2), seed)


  prepare_tagging_data_wo_label(tagging_data_dir + 'specific_question.txt',
                                '../../training/tagging/', 'specific_question',
                                (0.1, (0.1, 0.2), 0 if is_product else 0.2), seed)
  prepare_test_tagging_data_wo_label(tagging_test_data_file,
                                     '../../training/tagging/', 'specific_question',)
  add_label()
