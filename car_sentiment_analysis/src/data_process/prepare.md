# 数据预处理脚本说明

## prepare_v1_data.py（基础版本）
- 数据规模：1000条均衡数据
- 采样策略：简单随机采样
- 类别分布：四种情感类型各250条（强制均衡）
- 数据划分：训练集:验证集 = 8:2
- 输出文件：
  * train.csv：训练集数据（800条）
  * val.csv：验证集数据（200条）
  * all_data.csv：合并后的全部数据（1000条）
  * data_stats.json：数据统计信息

## prepare_v2_data.py（加权采样版本）
- 数据规模：3000条均衡数据
- 采样策略：基于评论质量的加权采样
  * 使用点赞数(like_count)和回复数(sub_comment_count)计算权重
  * 权重计算：weight = 1 + log(1 + 点赞数) + log(1 + 回复数)
  * 优先采样高质量评论
- 类别分布：四种情感类型各750条（强制均衡）
- 数据划分：训练集:验证集 = 8:2
- 新增特性：
  * 评论质量权重计算
  * 输出平均点赞和回复数统计
- 输出文件：
  * train.csv：训练集数据（2400条）
  * val.csv：验证集数据（600条）
  * all_data.csv：合并后的全部数据（3000条）
  * data_stats.json：包含质量统计的数据信息

## prepare_v3_data.py（全量数据版本）
- 数据规模：全量数据（6015条）
- 采样策略：无采样，使用全部数据
- 类别分布：保持原始分布不变
- 数据划分：训练集:验证集 = 8:2（保持类别分布一致）
- 新增特性：
  * 详细的类别百分比统计
  * 保证划分后训练集和验证集的类别分布与原始数据一致
- 输出文件：
  * train.csv：训练集数据（约4812条）
  * val.csv：验证集数据（约1203条）
  * all_data.csv：合并后的全部数据（6015条）
  * data_stats.json：包含详细分布统计的数据信息

## prepare_v4_data.py（大规模数据版本）
- 数据规模：30000条数据
- 采样策略：保持原始数据分布
- 数据划分：训练集:验证集:测试集 = 7:2:1
- 新增特性：
  * 增加测试集划分
  * 分层抽样确保各集合类别分布一致
  * 更详细的数据质量检查
  * 可读性更强的统计信息输出
- 数据质量检查：
  * 缺失值统计
  * 重复评论检测
  * 评论长度分析
  * 异常值识别
- 输出文件：
  * train.csv：训练集数据（70%）
  * val.csv：验证集数据（20%）
  * test.csv：测试集数据（10%）
  * data_stats.json：包含详细的数据统计信息
    - 每个数据集的样本总数
    - 类别分布（数量和百分比）
    - 数据质量指标
    - 可读性优化的分类标签

## 文件格式说明
- 所有CSV文件使用 `|` 作为分隔符
- 字段顺序：id|car_name|content|like_count|sub_comment_count|sentiment_analysis_results
- sentiment_analysis_results：
  * 0：负面评价
  * 1：正面评价
  * 2：中性评价
  * 3：无关评价

## 数据存储路径
- v1版本：data/experiments/v1_1000samples/
- v2版本：data/experiments/v2_3000samples/
- v3版本：data/experiments/v3_full_data/
- v4版本：data/experiments/v4_30000samples/