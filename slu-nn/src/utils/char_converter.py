import random


def word_pos_to_chars(words, pos):
  words = words.split(' ')
  pos = pos.split(' ')
  chars = []
  char_pos = []
  for z in zip(words, pos):
    w, p = z
    chars.append(list(w))
    char_pos.append([p] * len(list(w)))
  chars = [' '.join(i) for i in chars]
  chars = ' '.join(chars)
  char_pos = [' '.join(i) for i in char_pos]
  char_pos = ' '.join(char_pos)
  return chars, char_pos


def train_to_chars(input_file, output_file):
  f = open('../../training/domain/' + input_file, 'r')
  d = f.read().split('\n')

  r = []
  for l in d:
    l = l.split('\t')
    if len(l) != 3:
      continue
    char, pos = word_pos_to_chars(l[0], l[1])
    label = l[2].lower()
    r.append(char)
    r.append(pos)
    r.append(label)
  rand = []
  random.seed(42)
  for i in range(0, len(r) - 1, 3):
    rand.append((r[i], r[i + 1], r[i + 2]))
  random.shuffle(rand)
  r = []
  for l in rand:
    r.append(l[0])
    r.append(l[1])
    r.append(l[2])

  with open('../../data/char_cnn_classification/' + output_file, 'w+') as f:
    f.write('\n'.join(r))


def create_vocab(input_file):
  f = open('../../data/char_cnn_classification/' + input_file, 'r')
  d = f.read().split('\n')
  data = []
  for i in range(0, len(d) - 1, 3):
    data.append(d[i])

  vocab = {}
  if data is not None:
    for d in data:
      words = d.split()
      for w in words:
        if w in vocab:
          vocab[w] += 1
        else:
          vocab[w] = 1
  with open('word.vocab', "w+") as f:
    v = sorted(vocab, key=vocab.get, reverse=True)
    f.write("\n".join(v))


def validate_data(input_file):
  f = open('../../data/char_cnn_classification/' + input_file, 'r')
  d = f.read().split('\n')
  for i in range(0, len(d) - 1, 3):
    w = d[i].split(' ')
    p = d[i + 1].split(' ')
    if len(w) != len(p):
      print(w, len(w))
      print(p, len(p))


if __name__ == '__main__':
  # create_vocab()
  train_to_chars('train.src', 'train.in')
  train_to_chars('test.src', 'test.in')
  train_to_chars('dev.src', 'dev.in')
  validate_data('train.in')
  validate_data('test.in')
  validate_data('dev.in')
