# 汽车评论数据Spark处理模块

## 项目概述
本模块负责对汽车评论数据进行Spark处理和聚合，为后端API提供所需的统计数据。主要包括评论统计、情感分析统计、高频评论提取和关键词分析等功能。

## 数据源格式
### 原始数据表结构
每个品牌都有独立的数据表，表结构如下：
```sql
CREATE TABLE xxxx (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    car_id VARCHAR(6) COMMENT '汽车id',
    user_id VARCHAR(60) COMMENT '用户id',
    content LONGTEXT COMMENT '评论内容',
    like_count VARCHAR(255) COMMENT '点赞数',
    sub_comment_count VARCHAR(16) COMMENT '回复数'
    sentiment_analysis_results INT COMMENT '情感分析结果 0负面 1正面 2 中性 3 无关' 
);

```
Spark从原始数据中聚合得到下列信息
其中原始表xxxx是汽车英文名 我回给出列表 以及中文名列表

## 目标数据表设计

### 1. 车辆信息表 (car_info)
展示每款车型基础信息 共50行
```sql 
CREATE TABLE car_info (
    car_id INT NOT NULL COMMENT '车id',
    car_name VARCHAR(255) NOT NULL COMMENT '车名',
    car_type ENUM('SUV', 'MPV', '轿车') NOT NULL COMMENT '汽车类型',
    power_type ENUM('燃油', '电动') NOT NULL COMMENT '动力类型',
    origin_country VARCHAR(50) NOT NULL COMMENT '来源国家',
    PRIMARY KEY (car_id),
    FOREIGN KEY (car_id) REFERENCES car_evaluation_stats(car_id)
);
```

### 2. 评价统计表 (car_evaluation_stats)
```sql
每行数据 展示每款车型的 整体信息 共50行
CREATE TABLE car_evaluation_stats (
    car_id INT NOT NULL,
    car_name VARCHAR(255) NOT NULL,
    positive_evaluation_count INT NOT NULL COMMENT '正向评价数量',
    negative_evaluation_count INT NOT NULL COMMENT '负面评价数量',
    neutral_evaluation_count INT NOT NULL COMMENT '中性评价数量',
    total_evaluation_count INT NOT NULL COMMENT '总评价数',
    evaluation_rate DECIMAL(5,2) COMMENT '好评率',
    total_likes INT NOT NULL COMMENT '总点赞数',
    total_comments INT NOT NULL COMMENT '总评论数',
    updated_at TIMESTAMP COMMENT '更新时间',
    PRIMARY KEY (car_id)
);
```

### 3. 高频评论表 (car_comments)
每行数据展示：
    某款车型：点赞量/评论量/互动量的前30条评论 共30*3*50=4500 行
```sql
CREATE TABLE car_comments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    car_id INT NOT NULL,
    content TEXT NOT NULL,
    like_count INT NOT NULL,  COMMENT '点赞量',
    sub_comment_count INT NOT NULL, COMMENT '评论量'
    total_interaction INT NOT NULL COMMENT '互动总数',
    rank_type ENUM('like', 'comment', 'total') NOT NULL COMMENT '排名类型',
    rank_number INT NOT NULL COMMENT '排名',
    created_at TIMESTAMP,
    FOREIGN KEY (car_id) REFERENCES car_evaluation_stats(car_id),
    INDEX idx_car_rank (car_id, rank_type, rank_number)
);
```

### 4. 关键词表 (car_keywords)
每款车型的30个关键字  共30*50=1500
```sql
CREATE TABLE car_keywords (
    car_id INT NOT NULL,
    keyword VARCHAR(255) NOT NULL,
    frequency INT NOT NULL COMMENT '词频',
    rank INT NOT NULL COMMENT '排名',
    PRIMARY KEY (car_id, keyword),
    FOREIGN KEY (car_id) REFERENCES car_evaluation_stats(car_id)
);
```

### 5. 类别关键词表 (category_keywords)
根据类别 进行聚合 关键字 类别共5个 5*30 = 150行
```sql
CREATE TABLE category_keywords (
    category_description ENUM('燃油', '新能源', 'SUV', 'MPV', '轿车') NOT NULL COMMENT '类别描述',
    keyword VARCHAR(255) NOT NULL COMMENT '关键字',
    PRIMARY KEY (category_description, keyword)
);
```

## 业务需求

### 页面1：单车详情
1. 评论情感分布
   - 展示正面/负面/中性评论占比 
   - 数据来源：car_evaluation_stats表

2. 高频评论展示
   - 展示点赞量最高的前30条评论
   - 展示评论量最高的前30条评论
   - 展示总互动量最高的前30条评论
   - 数据来源：car_comments表

3. 关键词词云
   - 展示该车型的TOP30关键词
   - 数据来源：car_keywords表

### 页面2：汽车类别统计
1. 类别评价分布
   - 按不同类别（全部/燃油/新能源/SUV/MPV/轿车）展示评价分布
   - 数据来源：car_evaluation_stats + car_info表

2. TOP10车型展示
   - 按类别分别展示
   点赞量最高
   展示评论量
   展示总互动量最高
   - 前三款车型
   正面评价占比
   负面评价占比
   中性评价占比
    
   - 数据来源：car_evaluation_stats + car_info表

3. 类别词云
   - 展示类别分贝展示
   点赞量最高
   展示评论量
   展示总互动量最高
   前三款车型
   全部关键词词云
   - 数据来源：category_keywords表

4. 互动量分析
   - 展示不同类别车型的互动量（点赞/评论/总量）占比
   燃油/全部

   - 数据来源：car_evaluation_stats + car_info表

## Spark处理流程

### 1. 基础配置
```python
from pyspark.sql import SparkSession

class CarDataAggregator:
    def __init__(self):
        self.spark = SparkSession.builder \
            .appName("CarDataAggregation") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "10g") \
            .getOrCreate()
```

### 2. 数据处理流程
1. 数据读取和清洗
2. 评价统计聚合
3. 高频评论提取
4. 关键词分析
5. 结果存储

### 3. 更新策略
- 采用增量更新机制
- 每日凌晨定时更新
- 记录数据更新时间戳

## 性能优化建议

### 1. 数据存储优化
- 使用分区表存储原始数据
- 建立合适的索引
- 优化表结构和字段类型

### 2. Spark优化
- 合理设置分区数
- 使用缓存机制
- 优化JOIN操作

### 3. 更新策略优化
- 实现增量更新
- 批量处理
- 并行计算

## 监控和维护

### 1. 数据质量监控
- 数据完整性检查
- 异常值检测
- 更新状态监控

### 2. 性能监控
- 处理时间监控
- 资源使用监控
- 错误日志监控

### 3. 告警机制
- 处理失败告警
- 数据异常告警
- 性能问题告警

## 项目结构
```
src/spark/
├── aggregators/
│   ├── __init__.py
│   ├── evaluation_aggregator.py
│   ├── comment_aggregator.py
│   └── keyword_aggregator.py
├── utils/
│   ├── __init__.py
│   ├── db_utils.py
│   └── text_utils.py
├── config/
│   ├── __init__.py
│   └── config.py
└── main.py
```

## 下一步计划
1. 实现基础聚合功能
2. 添加数据质量检查
3. 优化性能
4. 实现监控系统
5. 完善文档和注释 