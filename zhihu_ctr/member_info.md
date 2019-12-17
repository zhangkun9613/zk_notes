| 属性   | 字段名 | 数据类型 | 类别数目 | 处理方式 | 分析 |
|-------| ------|------|------|------|------|
| 用户ID | user_id | str | 1931654 | None | ID,主键 |
| 性别 |gender|str| 3|onehot| male(21%)、female(15%),unknown(63%), 未知较多，不做填充|
| 访问频率| visit_freq|int|5|label| 按频率高低从0-4赋值 |
| 用户二分类特征A | binary_A | int | 2 | label | 60:40 |
| 用户二分类特征B | binary_B | int | 2 | label | 68:31 |
| 用户二分类特征C | binary_C | int | 2 | label | 97:3 |
| 用户二分类特征D | binary_D | int | 2 | label | 85:15 |
| 用户二分类特征E | binary_E | int | 2 | label | 95:5 |
| 用户多分类特征A | category_A | str | 2561 | onehot | 最大的类占50，其他的最高也仅3，大量类别仅有1个用户
| 用户多分类特征B | category_B | str | 291 | onehot | 最大的50，其次20，然后6
| 用户多分类特征C | category_C | str | 428 | onehot | 最大的11，其次7，然后6,分布较均匀
| 用户多分类特征D | category_D | str | 428 | onehot | 最大的40，其次6，然后4
| 用户多分类特征E | category_E | str | 2 | onehot | 91:8，以上多分类特征先不降维，分析相关性之后再进一步考虑
| 盐值 |salt_value|int| 95-890 | None | p_0.75 = 373 |
|用户关注的话题 | follow_topics | str | 80648 |label| 前250话题占据了大多数，共100000个话题，80000多个出现,3000多万数据,24%为空
|用户感兴趣话题 | interest_topics|str |0.0-inf|label|共1000多万条数据，格式为 topic:interest_index, 仅有20000多个话题，27%为空,(4.9242606, 0.07592721)