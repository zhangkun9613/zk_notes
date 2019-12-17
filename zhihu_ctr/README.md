比赛简介见 https://www.biendata.com/competition/zhihu2019 
#### 主要工作
1. 对应id特征，计算其出现的次数
2. 对应与历史有关的特征，仅计算某个时间区间内的，并计算频次或向量的均值
3. 由于训练测试数据按照时间顺序排列，测试数据的时间跨度为七天，因此训练时仅使用最后七天的邀请信息，采用之前时间的数据计算对应特征
#### 未完成工作
1. 将 gbdt 与 LR,deep fm进行融合,但是数据量过大，无法计算。
#### 具体训练细节：
* 不加入id_count , 0.69
* 加入user_id_count，question_id_count有明显提升,到 0.76
* 数据预归一化处理
* 加入 answer_prob 训练0.29，0.88，测试0.58,过拟合
* 加入 question_count,answer_count 也是过拟合
* 加入 top相关特征 训练0.74-0.79 提交0.77
* 加入回答的统计特征'num_agree_user','word_count_user','num_agree_question','word_count_question','question_answer_nums'，'uesr_answer_nums'，训练0.87 测试0.75
* 加入  均值填充nan 'num_agree_user','word_count_user','user_answer_nums'， 0.76-0.82， 0.46-0.35， 77.01 
* 同上 0均值 77.3
*  0均值 0.771 过拟合    'num_agree_user','word_count_user','user_answer_nums','num_agree_question' train's auc: 0.785558	train's binary_logloss: 0.465657 (0.8465617931917354, 0.33882461713047074)
*	mean   0.75 过拟合 'num_agree_user','word_count_user','num_agree_question','word_count_question' train's auc: 0.81656	train's binary_logloss: 0.465285 (0.8714833442600826, 0.316634370916078)
* mean num_agree_user num_agree_question 0.76

* 计算answer_df时不使用最近10天的数据，训练0.79-0.88 提交79.5
* 加入 answer_prob(-10), 使用（-10）数据 ，训练0.77-0.83 提交80.1
* 使用-10 76-82 79.58
* answer question prob （-10）0.77-0.83 78.45
* answer question prob （-20） 数据-10  0.76-0.83 79.8
* answer question prob （-20） 数据-20  0.78-0.86  79.8
* 加入 新特征 answer(-10) 数据-10 0.77-0.839 78.4
* 加入 新特征 answer(-20) 数据-10 0.77-0.831 80.0
* 去除 answer question 数据 -10 80.55

* retrain 前80.1 retrain后80.4
* 加入 category_columns, 使用默认参数，训练0.75-0.85 79.9
* 使用修改后的参数， 训练0.77-0.83 80.51
* 加入 topic_cosdi -10 同上 80.53 似乎不用加入category_columns
* 加入 topic_cosdis 全部数据 80.22
* 同上 去除 category columns, 训练 0.77-0.83 80.56
* 测试以下参数
* 数据天数 -14 77-84 80.57
* 数据天数 -7 76-83 80.63
* 数据天数 -10 80.64
*，预训练   7+7 1000,2000  80.80
* 7+7+7 2000 80.856
* 10 +7 2000 2000 80.85

* 加入 answer_prob -7 -7 82.454
* 加入 answer_prob -7 82.37
* 加入 answer_prob question  -7 82.40
* 加入 answer_prob question  -7 -7 -7 训练84.75 提交82.58
* 加入欧式距离 未归一化 ，使用category -7 82.21
* 加入欧式距离 归一化 ，-7 82.44
* 加入话题距离 -7 82.61
* 加入 category -7 82.43
* 加 user_df和question_df的idcount -7 82.61
* 加入标题信息 -7 82.71
