#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import random
import fasttext
import shutil
from model import SiameseLSTM
import argparse

def init_word_vectors_fasttext():
  fasttext_model = fasttext.load_model('../../data/fasttext_model/zh.bin')
  def word2vec(w):
    return fasttext_model[w]
  return word2vec

def init_pos_vectors():
  f = open('../../data/siamese_classification/pos.vocab', 'r')
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
def load_data(path, word2vec, pos2vec, is_train=False):
  data = open(path, 'r').read().split('\n')
  data = [i.split('\t') for i in data]
  data = [i for i in data if len(i) == 4]
  y = [1.0] * len(data)
  if is_train:
    neg_data = open('../../data/siamese_classification/train_neg_plus.txt', 'r').read().split('\n')
    neg_data = [i.split('\t') for i in neg_data]
    neg_data = [i for i in neg_data if len(i) == 4]
    for i in data:
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
    data += neg_data
    y += [0.0] * len(neg_data)
  x1_words = [i[0].split(' ') for i in data]
  x1_pos = [i[1].split(' ') for i in data]
  x1_wv, x1_pv, x1_seq_len = feature_prepare(x1_words, x1_pos, word2vec, pos2vec)
  x2_words = [i[2].split(' ') for i in data]
  x2_pos = [i[3].split(' ') for i in data]
  x2_wv, x2_pv, x2_seq_len = feature_prepare(x2_words, x2_pos, word2vec, pos2vec)
  x1 = [''.join(i) for i in x1_words]
  return list(zip(x1_wv, x1_pv, x1_seq_len, x2_wv, x2_pv, x2_seq_len, y, x1))

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

def train(output_folder):
  word2vec = init_word_vectors_fasttext()
  pos2vec = init_pos_vectors()
  data = load_data('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec, is_train=True)
  #test_data = load_data('../../data/siamese_classification/test.txt', word2vec, pos2vec)
  batch_size = 32
  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      model = SiameseLSTM(20, 300, 256, 0.0, batch_size, training=True)
      global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(0.001)

    grads_and_vars = optimizer.compute_gradients(model.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    output_dir = '../../output/{}'.format(output_folder)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary = tf.summary.scalar("accuracy", model.accuracy)
    summary = tf.summary.merge([loss_summary, acc_summary])
    checkpoint_prefix = os.path.join(output_dir, "model")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
    sess.run(tf.global_variables_initializer())


    for batch in batch_iter(data, batch_size, 20, shuffle=True):
      x1_wv, x1_pv, x1_seq_len, x2_wv, x2_pv, x2_seq_len, y, x1 = zip(*batch)
      feed_dict = {
        model.input_x1_wv: x1_wv,
        model.input_x1_pv: x1_pv,
        model.x1_seq_len: x1_seq_len,
        model.input_x2_wv: x2_wv,
        model.input_x2_pv: x2_pv,
        model.x2_seq_len: x2_seq_len,
        model.input_y: y,
        model.dropout_keep_prob: 0.7,
      }
      _, step, loss, accuracy, dist, sim, step_summary = sess.run(
            [tr_op_set, global_step, model.loss, model.accuracy, model.distance, model.temp_sim, summary], feed_dict)
      print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
      summary_writer.add_summary(step_summary, step)

    saver.save(sess, checkpoint_prefix, global_step=step)
    summary_writer.close()

def retrain():
  word2vec = init_word_vectors_fasttext()
  pos2vec = init_pos_vectors()
  data = load_data('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec, is_train=True)
  #test_data = load_data('../../data/siamese_classification/test.txt', word2vec, pos2vec)
  batch_size = 32 ; epoch = 2
  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      model = SiameseLSTM(20, 300, 256, 0.0, batch_size, training=True)
      global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(0.001)

    grads_and_vars = optimizer.compute_gradients(model.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    output_dir = '../../output/retrain_siamese'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    summary_writer = tf.summary.FileWriter(output_dir, sess.graph)
    loss_summary = tf.summary.scalar("loss", model.loss)
    acc_summary = tf.summary.scalar("accuracy", model.accuracy)
    summary = tf.summary.merge([loss_summary, acc_summary])
    checkpoint_prefix = os.path.join(output_dir, "model")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
    ckpt = tf.train.latest_checkpoint('../../output/siamese_classifier')
    saver.restore(sess, ckpt)

    for batch in batch_iter(data, batch_size, epoch, shuffle=True):
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

def test(output_folder):
  word2vec = init_word_vectors_fasttext()
  pos2vec = init_pos_vectors()
  label_data = load_label_data(word2vec, pos2vec)
  test_data = load_test_data('../../data/siamese_classification/train_plus.txt', word2vec, pos2vec)
  neg_data = load_test_data('../../data/siamese_classification/train_neg_plus.txt', word2vec, pos2vec)

  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      model = SiameseLSTM(20, 300, 256, 0.0, 1, training=False)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
    ckpt = tf.train.latest_checkpoint('../../output/{}'.format(output_folder))
    saver.restore(sess, ckpt)

    x2_wv, x2_pv, x2_seq_len, _ = zip(*label_data)
    result = []
    for d in test_data:
      # print(d[3])
      feed_dict = {
        model.input_x1_wv: [d[0]] * len(INTENTS),
        model.input_x1_pv: [d[1]] * len(INTENTS),
        model.x1_seq_len: [d[2]] * len(INTENTS),
        model.input_x2_wv: x2_wv,
        model.input_x2_pv: x2_pv,
        model.x2_seq_len: x2_seq_len
      }
      dist, sim = sess.run([model.distance, model.temp_sim], feed_dict)
      idx = np.argmin(dist, axis = 0)
      if dist.min() > 0.3:
        result.append(d[3] + '\t' + d[4] + '\t' + 'NEG')
      else:
        result.append(d[3] + '\t' + d[4] + '\t' + INTENTS[idx])
      if INTENTS[idx] != d[4] or dist.min() > 0.3:
          print(d[3])
          print(INTENTS[idx],dist.min())
    pre = ''
    print('neg\n')
    for d in neg_data:
      if d[3] != pre:
        pre = d[3]
      else:
        continue
      feed_dict = {
        model.input_x1_wv: [d[0]] * len(INTENTS),
        model.input_x1_pv: [d[1]] * len(INTENTS),
        model.x1_seq_len: [d[2]] * len(INTENTS),
        model.input_x2_wv: x2_wv,
        model.input_x2_pv: x2_pv,
        model.x2_seq_len: x2_seq_len
      }
      dist, sim = sess.run([model.distance, model.temp_sim], feed_dict)
      idx = np.argmin(dist, axis = 0)
      if dist.min() > 0.3:
        result.append(d[3] + '\t' + 'NEG'  + '\t' + 'NEG')
      else:
        result.append(d[3] + '\t' + 'NEG' + '\t' + INTENTS[idx])
      if dist.min() < 0.3:
          print(d[3])
          print(dist.min())
    with open('../../output/siamese_classifier/sia_result.txt','w+') as f:
        f.write('\n'.join(result))

def export(output_folder):
  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      model = SiameseLSTM(20, 300, 256, 0.0, 32, training=False)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    output_dir = '../../output/{}'.format(output_folder)
    ckpt = tf.train.latest_checkpoint(output_dir)
    saver.restore(sess, ckpt)
    export_dir = '../../export/compare_domain/1'
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    builder = tf.saved_model.builder.SavedModelBuilder('../../export/compare_domain/1')
    x1_wv_info = tf.saved_model.utils.build_tensor_info(model.input_x1_wv)
    x1_pv_info = tf.saved_model.utils.build_tensor_info(model.input_x1_pv)
    x1_seq_len_info = tf.saved_model.utils.build_tensor_info(model.x1_seq_len)
    x2_wv_info = tf.saved_model.utils.build_tensor_info(model.input_x2_wv)
    x2_pv_info = tf.saved_model.utils.build_tensor_info(model.input_x2_pv)
    x2_seq_len_info = tf.saved_model.utils.build_tensor_info(model.x2_seq_len)
    # output_info = tf.saved_model.utils.build_tensor_info(model.distance)
    out1_info = tf.saved_model.utils.build_tensor_info(model.out1)
    out2_info = tf.saved_model.utils.build_tensor_info(model.out2)
    prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
          'x1_wv': x1_wv_info,
          'x1_pv': x1_pv_info,
          'x1_seq_len': x1_seq_len_info,
          'x2_wv': x2_wv_info,
          'x2_pv': x2_pv_info,
          'x2_seq_len': x2_seq_len_info,
        },
        outputs={
          # 'outputs': output_info,
          'out1': out1_info,
          'out2': out2_info,
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
      )
    )
    init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
        'domain_compare': prediction_signature
      },
      legacy_init_op=init_op)
    builder.save()

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='train')
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--export', action='store_true', help='export')
parser.add_argument('--retrain', action='store_true', help='retrain')
parser.add_argument('--output', default='siamese_classifier', type=str, help='output_dir')
opt = parser.parse_args()


if __name__ == '__main__':
  if opt.train:
    train(opt.output)
  if opt.retrain:
    retrain()
  if opt.test:
    test(opt.output)
  if opt.export:
    export(opt.output)
