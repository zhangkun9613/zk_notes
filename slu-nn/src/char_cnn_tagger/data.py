#coding=utf-8
import numpy as np

input_dir = "../../data/slot_filling/"
output_dir = "../../data/char_cnn_tagger/"

sos = "<s>"
eos = "</s>"
unk = "<unk>"



def train_data_preprocess(inpath, outpath, mode, weight):

  def get_data(inpath):
    with open(inpath) as cin:
      data = cin.readlines()
      word_input = data[::4]
      pos_input = data[1::4]
      target_input = data[2::4]
      return list(zip(word_input, pos_input, target_input))

  def norm(weight):
    assert len(weight) == 3
    s = weight[0] + weight[1] + weight[2]
    return [w / s for w in weight]

  def write_data(data, outpath, mode):
    with open(outpath, mode) as cout:
      for word_input, pos_input, target_input in data:
        word_input = word_input.strip().split(' ')
        pos_input = pos_input.strip().split(' ')
        target_input = target_input.strip().split(' ')
        target_output = [" ".join([tag] * len(word)) for word, tag in zip(word_input, target_input)]
        pos_output = [" ".join([pos] * len(word)) for word, pos in zip(word_input, pos_input)]
        word_input = " ".join(list("".join(word_input)))
        target_output = " ".join(target_output)
        pos_output = " ".join(pos_output)
        cout.write(word_input + "\n")
        cout.write(pos_output + "\n")
        cout.write(target_output + "\n")

  weight = norm(weight)
  data = get_data(inpath)
  data = np.array(data)
  data_len = len(data)
  shuffle_indices = np.random.permutation(np.arange(data_len))
  data = data[shuffle_indices]
  train_len, dev_len, test_len = int(data_len * weight[0]), int(data_len * weight[1]), int(data_len * weight[2])
  train_data = data[0: train_len]
  dev_data = data[train_len: train_len + dev_len]
  test_data = data[train_len + dev_len: len(data)]
  write_data(train_data, outpath + "train.in", mode)
  write_data(dev_data, outpath + "dev.in", mode)
  write_data(test_data, outpath + "test.in", mode)

def create_vocabulary(data, outpath):
  vocab = dict([(sos, 1), (eos, 1), (unk, 1)])
  with open(outpath, "w") as cout:
    for line in data:
      line = line.strip().split(" ")
      line = filter(lambda x: x not in vocab, line)
      vocab.update(dict([(w, 1) for w in line]))
    cout.write("\n".join(vocab.keys()))

def vocab_preprocess(input_dir, output_dir):
  with open(input_dir + "train.in") as cin:
    data = cin.readlines()
    create_vocabulary(data[::3], output_dir + "vocabs/word.vocab")
    create_vocabulary(data[1::3], output_dir + "vocabs/pos.vocab")
    create_vocabulary(data[2::3], output_dir + "vocabs/target.vocab")
  with open(output_dir + "vocabs/char.vocab", "w") as cout:
    cout.write(sos + "\n")
    cout.write(eos + "\n")
    cout.write(unk + "\n")
    for i in range(26): cout.write(chr(ord('a') + i) + "\n")

if __name__ == "__main__":
  weight = [7, 2, 1]
  train_data_preprocess(input_dir + "media.txt", output_dir, "w", weight)
  train_data_preprocess(input_dir + "command.txt", output_dir, "a", weight)
  vocab_preprocess(output_dir, output_dir)
