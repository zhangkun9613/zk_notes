import tensorflow as tf

import fasttext
from crf_python import segment
from crf_python import segmentor_param_pb2

fasttext_model = None
seg = None
seg_parameters = None

def init_segmentor():
  global fasttext_model, seg, seg_parameters
  seg = segment.Segmentor()
  if not seg.init('./crf_python/libcrf_segmentor_interface.so', './segment_dict'):
    print("Failed to init the segmentor, please check the parameters.")
  seg_parameters = segmentor_param_pb2.SegmentorParameters()
  seg_parameters.keep_punc = False
  seg_parameters.need_pos = True
  seg_parameters.rec_long_entity = False
  seg_parameters.rec_ambiguous_entity = True
  seg_parameters.auto_rec_nr = True
  seg_parameters.recognizer_entity = True
  seg_parameters.need_term_weight = False

def sentence_to_words(sentence):
  word_node_list = seg.segment(str.encode(sentence), seg_parameters)
  return [w.word_info.word for w in word_node_list.word_node]

def generate_data():
  with tf.gfile.GFile('/work/ml/slu-rnn/data/Qiyi_intent/test/test.seq.in', mode='r') as f:
    sentences = f.read().splitlines()
  with tf.gfile.GFile('/work/ml/slu-rnn/data/Qiyi_intent/test/test.label', mode='r') as f:
    labels = f.read().splitlines()
  inputs = zip(sentences, labels)
  inputs = [s + '\t__label__' + l  for (s, l) in inputs]
  with tf.gfile.GFile('./train.txt', 'w+') as f:
    f.write("\n".join(inputs))

if __name__ == '__main__':
  classifier = fasttext.supervised('train.txt', 'tmp', label_prefix='__label__', epoch=500)
  with tf.gfile.GFile('/work/ml/slu-rnn/data/Qiyi_intent/test/test.seq.in', mode='r') as f:
    sentences = f.read().splitlines()
  labels = classifier.predict(sentences)
  outputs = zip(sentences, labels)
  outputs = [s + ' ' + l[0] for (s, l) in outputs]
  with tf.gfile.GFile('./predict.txt', 'w+') as f:
    f.write("\n".join(outputs))