#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import random
import fasttext
import time

from sia_model import SiameseLSTM
from model import SiameseLSTM as SIAMESE

def init_word_vectors_fasttext():
  fasttext_model = fasttext.load_model('../../data/fasttext_model/zh.bin')
  def word2vec(w):
    return fasttext_model[w]
  return word2vec

def init_pos_vectors():
  f = open('../../data/char_cnn_classification/vocabs/pos.vocab', 'r')
  rev_vocab = f.read().split('\n')
  vocab = {x: y for (y, x) in enumerate(rev_vocab)}
  print('pos vector initialization done')
  def pos2vec(p):
    ret = np.zeros(46, dtype=np.float32)
    np.put(ret, vocab[p], 1)
    return ret
  return pos2vec

def feature_prepare(words, pos, word2vec, pos2vec):
  words = [i[:20] for i in words]
  pos = [i[:20] for i in pos]
  seq_len = [len(i) for i in words]
  words = [i+['</s>']*(20-len(i)) for i in words]
  pos = [i+['</s>']*(20-len(i)) for i in pos]
  wv = [[word2vec(j) for j in i] for i in words]
  pv = [[pos2vec(j) for j in i] for i in pos]
  return wv, pv, seq_len

INTENTS = ['导演 查询', '演员 查询', '明星 关系 查询', '天气 查询', '股票 查询', '限行 查询', '内容 跳转', '演员 限定' ]
INTENT_POS = ['n v', 'n v', 'n n v', 'n v', 'n v', 'v v', 'n v', 'n v' ]
CLASSES = {
  '导演 查询': 0, '演员 查询': 1, '明星 关系 查询': 2, '天气 查询': 3, '股票 查询': 4, '限行 查询': 5, '内容 跳转': 6, '演员 限定': 7
}

def load_tridata(path,word2vec = None, pos2vec = None, is_train = True):
  data = open(path, 'r').read().split('\n')
  data = [i.split('\t') for i in data]
  data = [i for i in data if len(i) == 4]
  datas = {}
  for val in CLASSES.values():
    datas[val] = []
  for i in data:
    position = INTENTS.index(i[2])
    datas[position].append(i)
  if is_train:
    neg_data = open('../../data/siamese_classification/train_neg_plus.txt', 'r').read().split('\n')
    neg_data = [i.split('\t') for i in neg_data]
    neg_data = [i for i in neg_data if len(i) == 4]
    all_neg = []
    neg_sent = []
    for pos in range(len(neg_data)):
      if neg_data[pos][0] not in neg_sent:
        all_neg.append(neg_data[pos])
        neg_sent.append(neg_data[pos][0])
    #datas[-1] = all_neg
    all_neg = ['\t'.join(i[0:2]) for i in all_neg]
    with open('neg_data.txt', 'w+') as f:
      f.write('\n'.join(all_neg))
    x1 = []
    x2 = []
    x = []
    nums_same = 300
    nums_diff = 300
    classes = list(datas.keys())
    print(classes)
    y = []
    for cla in classes:
      for i in range(nums_same):
        sample1 = random.choice(datas[cla])
        sample2 = random.choice(datas[cla])
        x1.append(sample1)
        x2.append(sample2)
        x.append(str(sample1[0]) + '\t' + str(sample2[0]))
      y += [1.0] * nums_same
      for i in range(nums_diff):
        sample1 = random.choice(datas[cla])
        while True:
          cla_neg = random.choice(classes)
          if cla != cla_neg:
            break
        sample2 = random.choice(datas[cla_neg])
        x1.append(sample1)
        x2.append(sample2)
      y += [0.0] * nums_same
    #     x.append(str(sample1[0]) + '\t' + str(sample2[0]))
    # with open('temp.txt', 'w+') as f:
    #   f.write('\n'.join(x))
    x1_words = [i[0].split(' ') for i in x1]
    x1_pos = [i[1].split(' ') for i in x1]
    x1_wv, x1_pv, x1_seq_len = feature_prepare(x1_words, x1_pos, word2vec, pos2vec)
    x2_words = [i[0].split(' ') for i in x2]
    x2_pos = [i[1].split(' ') for i in x2]
    x2_wv, x2_pv, x2_seq_len = feature_prepare(x2_words, x2_pos, word2vec, pos2vec)
    return list(zip(x1_wv, x1_pv, x1_seq_len, x2_wv, x2_pv, x2_seq_len, y, x1))
  else:
    test_data = {}
    classes = list(datas.keys())
    for cla in classes:
      x = datas[cla]
      x_words = [i[0].split(' ') for i in x]
      x_pos = [i[1].split(' ') for i in x]
      x_wv, x_pv, x_seq_len = feature_prepare(x_words, x_pos, word2vec, pos2vec)
      test_data[cla] = list(zip(x_wv, x_pv, x_seq_len))
    return test_data

def load_twodata(path, word2vec=None, pos2vec=None, is_train=True):
  data = open(path, 'r').read().split('\n')
  data = [i.split('\t') for i in data]
  pos_data = [i for i in data if len(i) == 4]
  y = []
  data = []
  x =[]
  neg_data = open('../../data/siamese_classification/train_neg_plus.txt', 'r').read().split('\n')
  neg_data = [i.split('\t') for i in neg_data]
  neg_data = [i for i in neg_data if len(i) == 4]
  num_same = 3000
  num_diff = 5000
  if is_train:
    for file_name in ['media.txt','command.txt','question.txt']:
      domain_data = open('../../data/domain_classification/' + file_name, 'r').read().split('\n')
      domain_neg = []
      for i in range(0,len(domain_data),2):
        domain_neg.append([domain_data[i],domain_data[i+1]])
      domain_neg = random.choices(domain_neg,2000)
      neg_data.extend(domain_neg)

  for i in range(num_same):
    if i > num_same*0.5:
      sample1 = random.choice(pos_data)
      sample2 = random.choice(pos_data)
      d = [sample1[0], sample1[1], sample2[0], sample2[1]]
    else:
      sample1 = random.choice(neg_data)
      sample2 = random.choice(neg_data)
      d = [sample1[0], sample1[1], sample2[0], sample2[1]]
    x.append(str(sample1[0]) + '\t' + str(sample2[0]))
    data.append(d)
  y += [1.0]*num_same
  for i in range(num_diff):
    sample1 = random.choice(pos_data)
    sample2 = random.choice(neg_data)
    d = [sample1[0], sample1[1], sample2[0], sample2[1]]
    x.append(str(sample1[0]) + '\t' + str(sample2[0]))
    data.append(d)
  y += [0.0]*num_diff
    #     x.append(str(sample1[0]) + '\t' + str(sample2[0]))
    # with open('temp.txt', 'w+') as f:
    #   f.write('\n'.join(x))
  x1_words = [i[0].split(' ') for i in data]
  x1_pos = [i[1].split(' ') for i in data]
  x1_wv, x1_pv, x1_seq_len = feature_prepare(x1_words, x1_pos, word2vec, pos2vec)
  x2_words = [i[2].split(' ') for i in data]
  x2_pos = [i[3].split(' ') for i in data]
  x2_wv, x2_pv, x2_seq_len = feature_prepare(x2_words, x2_pos, word2vec, pos2vec)
  return list(zip(x1_wv, x1_pv, x1_seq_len, x2_wv, x2_pv, x2_seq_len, y, x))


def load_prob_data(path, word2vec=None, pos2vec=None, is_train=True):
  pos_data = open(path, 'r').read().split('\n')
  pos_data = [i.split('\t') for i in pos_data]
  pos_data = [i for i in pos_data if len(i) == 4]
  neg_data = open('../../data/siamese_classification/train_neg_plus.txt', 'r').read().split('\n')
  neg_data = [i.split('\t') for i in neg_data]
  neg_data = [i for i in neg_data if len(i) == 4]
  for file_name in ['media.txt', 'command.txt', 'question.txt']:
    domain_data = open('../../data/domain_classification/' + file_name, 'r').read().split('\n')
    domain_neg = []
    for i in range(0, len(domain_data), 2):
      domain_neg.append([domain_data[i], domain_data[i + 1]])
    if is_train and file_name != 'command.txt':
      domain_neg = random.sample(domain_neg, 10000)
    neg_data.extend(domain_neg)
  if is_train:
    pos_data = pos_data * 2
  data = pos_data + neg_data
  y = [1.0] * len(pos_data) + [0.0] * len(neg_data)
  x1_words = [i[0].split(' ') for i in data]
  x1_pos = [i[1].split(' ') for i in data]
  x1_wv, x1_pv, x1_seq_len = feature_prepare(x1_words, x1_pos, word2vec, pos2vec)
  x1 = [''.join(i) for i in x1_words]
  return list(zip(x1_wv, x1_pv, x1_seq_len, y, x1))

def load_data(path, word2vec=None, pos2vec=None, is_prob=True):
  pos_data = open(path, 'r').read().split('\n')
  pos_data = [i.split('\t') for i in pos_data]
  pos_data = [i for i in pos_data if len(i) == 4]
  neg_data = open('../../data/siamese_classification/train_neg_plus.txt', 'r').read().split('\n')
  neg_data = [i.split('\t') for i in neg_data]
  neg_data = [i for i in neg_data if len(i) == 4]
  if is_prob:
    data = pos_data + neg_data
    y = [1.0] * len(pos_data) + [0.0] * len(neg_data)
  else:
    for i in range(len(pos_data)):
      pos_data[i][2] = '特殊 意图'
      pos_data[i][3] = 'a n'
    y = [1.0] * len(pos_data)
    data = []
    data.extend(pos_data)
    for i in pos_data:
      d = [i[0], i[1], '普通 提问', 'a n']
      data.append(d)
    y += [0.0] * len(pos_data)
    for i in range(len(neg_data)):
      neg_data[i][2] = '普通 提问'
      neg_data[i][3] = 'a n'
    data.extend(neg_data)
    y += [1.0] * len(neg_data)
    for i in neg_data:
      d = [i[0], i[1], '特殊 意图', 'a n']
      data.append(d)
    y += [0.0] * len(neg_data)
  x1_words = [i[0].split(' ') for i in data]
  x1_pos = [i[1].split(' ') for i in data]
  x1_wv, x1_pv, x1_seq_len = feature_prepare(x1_words, x1_pos, word2vec, pos2vec)
  x2_words = [i[2].split(' ') for i in data]
  x2_pos = [i[3].split(' ') for i in data]
  x2_wv, x2_pv, x2_seq_len = feature_prepare(x2_words, x2_pos, word2vec, pos2vec)
  x1 = [''.join(i) for i in x1_words]
  return list(zip(x1_wv, x1_pv, x1_seq_len, x2_wv, x2_pv, x2_seq_len, y, x1))

def load_kfold_data(path, word2vec, pos2vec):
  data = open(path, 'r').read().split('\n')
  data = [i.split('\t') for i in data]
  data = [i for i in data if len(i) == 4]
  random.shuffle(data)
  val_data = data[:int(len(data)*0.1)]
  train_data = data[int(len(data)*0.9):]
  y = [1.0] * len(train_data)
  neg_data = open('../../data/siamese_classification/train_neg_plus.txt', 'r').read().split('\n')
  neg_data = [i.split('\t') for i in neg_data]
  neg_data = [i for i in neg_data if len(i) == 4]
  for i in train_data:
    while True:
      position = INTENTS.index(i[2])
      if position < 3 or position >= 6:
        t = random.choice(list(range(0, 3))+list(range(6, 8)))
      else:
        t = random.choice(range(3, 6))
      if INTENTS[t] != i[2]:
        break
    d = [i[0], i[1], INTENTS[t], INTENT_POS[t]]
    neg_data.append(d)
  train_data += neg_data
  y += [0.0] * len(neg_data)
  x1_words = [i[0].split(' ') for i in train_data]
  x1_pos = [i[1].split(' ') for i in train_data]
  x1_wv, x1_pv, x1_seq_len = feature_prepare(x1_words, x1_pos, word2vec, pos2vec)
  x2_words = [i[2].split(' ') for i in train_data]
  x2_pos = [i[3].split(' ') for i in train_data]
  x2_wv, x2_pv, x2_seq_len = feature_prepare(x2_words, x2_pos, word2vec, pos2vec)
  x1 = [''.join(i) for i in x1_words]
  train_data = list(zip(x1_wv, x1_pv, x1_seq_len, x2_wv, x2_pv, x2_seq_len, y, x1))
  x1_words = [i[0].split(' ') for i in val_data]
  x1_pos = [i[1].split(' ') for i in val_data]
  x1_wv, x1_pv, x1_seq_len = feature_prepare(x1_words, x1_pos, word2vec, pos2vec)
  x1_label = [i[2] for i in val_data]
  x1 = [''.join(i) for i in x1_words]
  val_data = list(zip(x1_wv, x1_pv, x1_seq_len, x1, x1_label))
  return train_data, val_data

def load_test_data(path, word2vec, pos2vec):
  data = open(path, 'r').read().split('\n')
  data = [i.split('\t') for i in data]
  data = [i for i in data if len(i) == 4]
  x1_words = [i[0].split(' ') for i in data]
  x1_pos = [i[1].split(' ') for i in data]
  x1_wv, x1_pv, x1_seq_len = feature_prepare(x1_words, x1_pos, word2vec, pos2vec)
  x1 = [''.join(i) for i in x1_words]
  x1_label = [i[2] for i in data]
  return list(zip(x1_wv, x1_pv, x1_seq_len, x1, x1_label))

def load_label_data(word2vec, pos2vec):
  label_words = [i.split(' ') for i in INTENTS]
  label_pos = [i.split(' ') for i in INTENT_POS]
  label_wv, label_pv, label_seq_len = feature_prepare(label_words, label_pos, word2vec, pos2vec)
  return list(zip(label_wv, label_pv, label_seq_len, INTENTS))

def batch_iter(data, batch_size, num_epochs, shuffle=True):
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

def train(is_one = True):
  word2vec = init_word_vectors_fasttext()
  pos2vec = init_pos_vectors()
 # data = load_data('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec, is_train=True)
  if is_one:
    data = load_data('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec, is_prob=False)
  else:
    data = load_twodata('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec, is_train=True)
 #test_data = load_data('../../data/siamese_classification/test.txt', word2vec, pos2vec)
  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      model = SIAMESE(20, 300, 256, 0.0, 32, training=True)
      global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(0.001)

    grads_and_vars = optimizer.compute_gradients(model.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    output_dir = '../../output/kfold'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary = tf.summary.scalar("accuracy", model.accuracy)
    summary = tf.summary.merge([loss_summary, acc_summary])
    checkpoint_prefix = os.path.join(output_dir, "model")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    sess.run(tf.global_variables_initializer())


    for batch in batch_iter(data, 32, 30, shuffle=True):
      x1_wv, x1_pv, x1_seq_len, x2_wv, x2_pv, x2_seq_len, y, x1 = zip(*batch)
      feed_dict = {
        model.input_x1_wv: x1_wv,
        model.input_x1_pv: x1_pv,
        model.x1_seq_len: x1_seq_len,
        model.input_x2_wv: x2_wv,
        model.input_x2_pv: x2_pv,
        model.x2_seq_len: x2_seq_len,
        model.input_y: y,
        model.dropout_keep_prob: 0.6,
      }
      _, step, loss, accuracy, dist, sim, step_summary = sess.run(
            [tr_op_set, global_step, model.loss, model.accuracy, model.distance, model.temp_sim, summary], feed_dict)
      print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
      summary_writer.add_summary(step_summary, step)

    saver.save(sess, checkpoint_prefix, global_step=step)
    summary_writer.close()

def train_test_prob(is_train=True, is_test=True):
  from sia_model import LstmProb
  word2vec = init_word_vectors_fasttext()
  pos2vec = init_pos_vectors()
  data = load_prob_data('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec, is_train=True)
  output_dir = '../../output/tri_siamese'
  batch_size = 64
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      model = LstmProb(20, 300, 256, 0.0, 32, True, 2)
    if is_train:
      global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(0.001)
      grads_and_vars = optimizer.compute_gradients(model.loss)
      tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
      summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
      loss_summary = tf.summary.scalar("loss", model.loss)
      acc_summary = tf.summary.scalar("accuracy", model.accuracy)
      summary = tf.summary.merge([loss_summary, acc_summary])
      checkpoint_prefix = os.path.join(output_dir, "model")
      saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
      sess.run(tf.global_variables_initializer())

      for batch in batch_iter(data, batch_size, 20, shuffle=True):
        x1_wv, x1_pv, x1_seq_len, y, x1 = zip(*batch)
        y = np.array(y).astype(np.int64)
        feed_dict = {
          model.input_x1_wv: x1_wv,
          model.input_x1_pv: x1_pv,
          model.x1_seq_len: x1_seq_len,
          model.input_y: y,
          model.dropout_keep_prob: 0.6,
        }
        _, step, loss, accuracy, step_summary = sess.run(
          [tr_op_set, global_step, model.loss, model.accuracy, summary], feed_dict)
        print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
        summary_writer.add_summary(step_summary, step)
      saver.save(sess, checkpoint_prefix, global_step=step)
      summary_writer.close()
    if is_test:
      test_data = load_prob_data('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec, is_train=False)
      saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
      ckpt = tf.train.latest_checkpoint(output_dir)
      saver.restore(sess, ckpt)
      for d in test_data:
        x1_wv, x1_pv, x1_seq_len, y, x1 = d
        feed_dict = {
          model.input_x1_wv: [x1_wv],
          model.input_x1_pv: [x1_pv],
          model.x1_seq_len: [x1_seq_len],
          model.dropout_keep_prob: 1,
        }
        prob = sess.run([model.prob], feed_dict)
        y_predict = np.argmax(prob)
        if y_predict != y:
          print(prob, x1)


def test():
  word2vec = init_word_vectors_fasttext()
  pos2vec = init_pos_vectors()
  label_data = load_label_data(word2vec, pos2vec)
  test_data = load_test_data('../../data/siamese_classification/test_plus.txt', word2vec, pos2vec)
  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      model = SiameseLSTM(20, 300, 256, 0.0, 1, training=False)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
    ckpt = tf.train.latest_checkpoint('../../output/siamese_classifier')
    saver.restore(sess, ckpt)

    x2_wv, x2_pv, x2_seq_len, _ = zip(*label_data)
    for d in test_data:
      start = time.time()
      print(d[3])
      feed_dict = {
        model.input_x1_wv: [d[0]] * len(INTENTS),
        model.input_x1_pv: [d[1]] * len(INTENTS),
        model.x1_seq_len: [d[2]] * len(INTENTS),
        model.input_x2_wv: x2_wv,
        model.input_x2_pv: x2_pv,
        model.x2_seq_len: x2_seq_len
      }
      nums = 2
      feed_dict = {
        model.input_x1_wv: [d[0]]*nums,
        model.input_x1_pv: [d[1]]*nums,
        model.x1_seq_len: [d[2]]*nums,
        model.input_x2_wv: [x2_wv[0]]*nums,
        model.input_x2_pv: [x2_pv[0]]*nums,
        model.x2_seq_len: [x2_seq_len[0]]*nums
      }
      dist, sim = sess.run([model.distance, model.temp_sim], feed_dict)
      end = time.time()
      print('time:', end - start)
      print(dist)
      idx = np.argmin(dist, axis = 0)
      if dist[idx] >= 0.4:
        print('unknown')
      else:
        print(INTENTS[idx])

def test_two(is_one = True):
  word2vec = init_word_vectors_fasttext()
  pos2vec = init_pos_vectors()
  if is_one:
    test_data = load_data('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec)
  else:
    test_data = load_twodata('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec, is_train=False)
  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      model = SIAMESE(20, 300, 256, 0.0, 1, training=False)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
    ckpt = tf.train.latest_checkpoint('../../output/kfold')
    saver.restore(sess, ckpt)

    for d in test_data:
      # print(d[3])
      feed_dict = {
        model.input_x1_wv: [d[0]],
        model.input_x1_pv: [d[1]],
        model.x1_seq_len: [d[2]],
        model.input_x2_wv: [d[3]],
        model.input_x2_pv: [d[4]],
        model.x2_seq_len: [d[5]]
      }
      y = d[6]
      dist, sim = sess.run([model.distance, model.temp_sim], feed_dict)
      if (y==1 and dist>0.5) or (y==0 and dist<0.5):
          print(d[-1], dist, y)

def test_sia(k=10):
  word2vec = init_word_vectors_fasttext()
  pos2vec = init_pos_vectors()
  all_data = load_test_data('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec)
  neg_data = load_test_data('../../data/siamese_classification/train_neg_plus.txt', word2vec, pos2vec)
  all_vec = []
  labels = []
  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      model = SiameseLSTM(20, 300, 256, 0.0, 32, training=False)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    ckpt = tf.train.latest_checkpoint('../../output/tri_siamese/')
    saver.restore(sess, ckpt)
    for d in all_data:
      feed_dict = {
        model.input_x1_wv: [d[0]],
        model.input_x1_pv: [d[1]],
        model.x1_seq_len: [d[2]]
      }
      outvec = sess.run([model.out1], feed_dict)
      all_vec.extend(outvec)
      labels.append(INTENTS.index(d[4]))
    all_vec = np.vstack(all_vec)
    labels = np.array(labels)
    # outs = np.hstack([all_vec, labels.T])
    # np.save('vec.npy', outs)
    count = []
    num = 0
    for d in all_data:
      feed_dict = {
        model.input_x1_wv: [d[0]],
        model.input_x1_pv: [d[1]],
        model.x1_seq_len: [d[2]]
      }
      #outvec's shape is (1,1,256)
      outvec = sess.run([model.out1], feed_dict)
      dis = np.square(all_vec-outvec[0]).sum(axis=1).squeeze()
      votes = labels[np.argsort(dis)[1:k]]
      votes = np.bincount(votes)
      predict = votes.argmax()
      count.append([votes.max(), predict])
      if d[4] != INTENTS[predict]:
        print(d[3], d[4], INTENTS[predict])
        num += 1
    print(num)
    count = np.array(count).T
    print(count[0].max())
    count = np.bincount(count[0])
    print(count)

def test_neg(k=7):
  word2vec = init_word_vectors_fasttext()
  pos2vec = init_pos_vectors()
  pos_data = load_test_data('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec)
  neg_data = load_test_data('../../data/siamese_classification/train_neg_plus.txt', word2vec, pos2vec)
  test_data = load_tridata('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec, is_train=False)
  labels = []
  new = []
  for i in range(1,len(neg_data)):
    if neg_data[i][3] != neg_data[i-1][3]:
      new.append(neg_data[i])
  neg_data = new
  test_vecs = {}
  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      model = SIAMESE(20, 300, 256, 0.0, 32, training=False)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    ckpt = tf.train.latest_checkpoint('../../output/siamese_classifier/')
    saver.restore(sess, ckpt)
    print(test_data.keys())
    all_time = 0
    all = []
    spos = 0
    position = {}
    for cla in test_data.keys():
      all_vec = []
      for d in test_data[cla]:
        feed_dict = {
          model.input_x1_wv: [d[0]],
          model.input_x1_pv: [d[1]],
          model.x1_seq_len: [d[2]]
        }
        outvec = sess.run([model.out1], feed_dict)
        all.extend(outvec)
      labels.extend([cla]*len(test_data[cla]))
      position[cla] = [spos, spos+len(test_data[cla])]
      spos += len(test_data[cla])
    labels = np.array(labels).reshape([-1,1])
    all = np.vstack(all)
    outs = np.hstack([all, labels])
    np.save('vec.npy', outs)
    nums = 0 ; tag = 1
    dis = np.zeros([8, 1])
    all_data = pos_data
    pos_nums = len(pos_data)
    all_data.extend(neg_data)
    for i in range(len(all_data)):
      d = all_data[i]
      start = time.time()
      feed_dict = {
        model.input_x1_wv: [d[0]],
        model.input_x1_pv: [d[1]],
        model.x1_seq_len: [d[2]]
      }
      #inputwv shape is [20,300]
      #outvec's shape is (1,1,256)
      outvec = sess.run([model.out1], feed_dict)
      out1 = outvec[0].reshape([1, -1])

      out2 = all
      distance = np.sqrt(np.sum(np.square(out1 - out2), 1))
      distance = distance / (np.sqrt(np.sum(np.square(out1), 1)) + np.sqrt(np.sum(np.square(out2), 1)))
      for cla in test_data.keys():
        s = position[cla][0];  e = position[cla][1]
        dis[cla] = distance[s:e].mean()
      votes = labels[np.argsort(distance)[1:k]].reshape([-1])
      votes = np.bincount(votes)
      predict = votes.argmax()

      # predict = dis.argmin()
      end = time.time()
      all_time += end - start
      if tag:
        all_time = 0
        tag = 0
      if i > pos_nums:
        if dis[predict] < 0.3:
          print(d[3], d[4], INTENTS[predict])
          print(dis[predict])
          nums+=1
      else:
        if d[4] != INTENTS[predict] or dis[predict] > 0.3:
          print(d[3], d[4], INTENTS[predict])
          print(dis[predict])
          nums+=1
    print(nums)
    print(all_time/len(neg_data))


def test_seperate():
  word2vec = init_word_vectors_fasttext()
  pos2vec = init_pos_vectors()
  label_data = load_label_data(word2vec, pos2vec)
  test_data = load_test_data('../../data/siamese_classification/test_plus.txt', word2vec, pos2vec)
  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      model = SiameseLSTM(20, 300, 256, 0.0, 1, training=False)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
    ckpt = tf.train.latest_checkpoint('../../output/siamese_classifier')
    saver.restore(sess, ckpt)

    x2_wv, x2_pv, x2_seq_len, _ = zip(*label_data)
    feed_dict = {
      model.input_x1_wv: x2_wv,
      model.input_x1_pv: x2_pv,
      model.x1_seq_len: x2_seq_len,
      model.input_x2_wv: x2_wv,
      model.input_x2_pv: x2_pv,
      model.x2_seq_len: x2_seq_len
    }
    out2 = sess.run([model.out2], feed_dict)
    out2 = out2[0]
    out2 = np.array(out2)
    np.save("label_vel.npy", out2)
    nums = 8
    for d in test_data:
      start = time.time()
      print(d[3])
      feed_dict = {
        model.input_x1_wv: [d[0]]*nums,
        model.input_x1_pv: [d[1]]*nums,
        model.x1_seq_len: [d[2]]*nums,
        model.input_x2_wv: x2_wv,
        model.input_x2_pv: x2_pv,
        model.x2_seq_len: x2_seq_len
      }
      out1, dists = sess.run([model.out1, model.distance], feed_dict)
      out1 = out1[0].reshape([1, 256])
      distance = np.sqrt(np.sum(np.square(out1-out2), 1))
      distance = distance/(np.sqrt(np.sum(np.square(out1), 1)) + np.sqrt(np.sum(np.square(out2), 1)))
      end = time.time()
      print('time:', end - start)
      idx = np.argmin(distance, axis = 0)
      print(distance)
      print(dists)
      if distance[idx] >= 0.3:
        print('unknown')
      else:
        print(INTENTS[idx])

def test_unknown(file_name):
  word2vec = init_word_vectors_fasttext()
  pos2vec = init_pos_vectors()
  label_data = load_label_data(word2vec, pos2vec)
  x2_wv, x2_pv, x2_seq_len, _ = zip(*label_data)
  with open('../../data/domain_classification/{}.txt'.format(file_name)) as f:
    data = f.readlines()
  x1_words = []
  x1_pos = []
  for i in range(len(data)):
    if i%2==0:
      x1_words.append(data[i].strip().split(' '))
    else:
      x1_pos.append(data[i].strip().split(' '))
  x1_wv, x1_pv, x1_seq_len = feature_prepare(x1_words, x1_pos, word2vec, pos2vec)
  test_data = list(zip(x1_wv, x1_pv, x1_seq_len, x1_words))

  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      model = SIAMESE(20, 300, 256, 0.0, 1, training=False)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
    ckpt = tf.train.latest_checkpoint('../../output/retrain_siamese')
    saver.restore(sess, ckpt)

  result = []
  count = [0]*8
  for d in test_data:
    feed_dict = {
      model.input_x1_wv: [d[0]] * len(INTENTS),
      model.input_x1_pv: [d[1]] * len(INTENTS),
      model.x1_seq_len: [d[2]] * len(INTENTS),
      model.input_x2_wv: x2_wv,
      model.input_x2_pv: x2_pv,
      model.x2_seq_len: x2_seq_len
    }
    dist, sim = sess.run([model.distance, model.temp_sim], feed_dict)
    index = dist.argmin(axis=0)
    if dist.min() < 0.3:
      result.append('\t'.join([''.join(d[3]), str(dist.min()),INTENTS[index]]))
      count[index] += 1
  with open('bad.txt','a') as f:
    f.write(file_name + '\n' + '\n'.join(result) + '\n')
  with open('badre.txt','a+') as f:
    count = [str(i) for i in count]
    f.write(' '.join(count))

if __name__ == '__main__':
#  train()
#  test_two()
#  test_sia(10)
  # test_neg()
#   test()
  #export()
#  train_k_fold()
# test_seperate()
  # train_test_prob()
  #train_test_prob(is_train=False)
  with open('bad.txt','w+') as f:
    f.write('')
  #for i in ['command']:
  for i in ['command','unknown','media','question']:
    test_unknown(i)
