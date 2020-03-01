INTENTS = ['导演 查询', '演员 查询', '明星 关系 查询', '天气 查询', '股票 查询', '限行 查询', '内容 跳转', '演员 限定', 'NEG']
DOMAINS = ['media', 'question', 'unknown', 'command']
def analyze_reslut(type = 'sia', all = True):
  if all or type == 'sia':
    with open('../output/siamese_classifier/sia_result.txt') as f:
      result = f.readlines()
    result = [i.strip().split('\t')[1:] for i in result]
    for pos in INTENTS:
      calcu_pr(pos, result)

  if all or type == 'domain':
    with open('../output/char_cnn_classification/result.txt') as f:
      result = f.readlines()
    result = [i.strip().split('\t')[1:] for i in result]
    for pos in DOMAINS:
      calcu_pr(pos, result)

  if all or type == 'tagger':
    for domain in DOMAINS:
      if domain == 'unknown':
          continue
      with open('../output/tagging_{}/tagging.result'.format(domain)) as f:
        result = f.readlines()
      with open('../training/tagging/label_{}.vocab'.format(domain)) as f:
        intents = f.readlines()[3:]
        intents = [i.strip() for i in intents]
      result = [i.strip().split(' ') for i in result]
      intent_result = []
      acc = 0
      for i in range(1,len(result), 3):
        tag_label = result[i][0:-3]
        tag_predict = result[i+1][0:-2]
        if tag_label == tag_predict:
          acc += 1
        intent_label = result[i][-3].strip()
        intent_predict = result[i+1][-2]
        intent_result.append([intent_label, intent_predict])
      for pos in intents:
        calcu_pr(pos, intent_result)
      print(domain + '\t' + 'acc:' + str(3* acc/len(result)))


def calcu_pr(pos, result):
  tp, tn, fp, fn = [0, 0, 0, 0]
  for label, predict in result:
    if label == pos:
      if predict == pos:
        tp += 1
      else:
        fp += 1
    else:
      if predict == pos:
        fn += 1
      else:
        tn += 1
  print(pos + '\t' + 'precision:' + str(tp / (tp + fp)) + '\t' + 'recall:' + str(tp / (tp + fn)))

if __name__ == '__main__':
    analyze_reslut()
