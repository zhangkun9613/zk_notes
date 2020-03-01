# 更新流程

### siamese classifier  

功能：预测是否为特定问题意图中的一种

训练过程：直接运行src/siamese_classifier/main.py

### char_cnn_classifier

功能：若不是特定问题，预测是否为command, media, question中的一类

训练过程：

1. 预处理:  src/utils/data_utils.py
2. 预处理：src/utils/char_converter.py, 注意在生成的word.vocab中加上`<unk>`,` </s>`,` <s>`, 将label_xxx.vocab 合并到 tag_xxx.vocab
3. src/char_cnn_classifier/main.py

### tagger

功能：分别训练specific_question, command, media, question四个模型， 为语句打上标签和具体类别

训练过程：

1. 预处理：src/utils/data_utils.py
2. src/tagger/main.py

### 模型部署

1. 生成的模型都在/export下， 将模型上传至10.153.169.115的/data/model_serving/export/下
2. 运行tensorflow_model_server --port=9001 --model_config_file=/data/model_serving/model_serving.conf &

