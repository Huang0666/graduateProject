# 汽车评论情感分析实验

本项目针对汽车评论进行多维度情感分析，通过三次渐进式训练提升模型性能。

## 实验流程

### 统计训练样本分类占比情况

### 规划每次训练所用各个分类样本占比
src/data_process
└──sentiment_stats.py

### 准备数据
1将数据按照规划占比划分训练集和测试集
保存数据统计信息
src/data_process
├──prepare_v1_data.py
├──prepare_v2_data.py
└──prepare_v3_data.py


### 第一次训练 (1000条数据)

#### 1. 数据准备阶段
```
输入数据：all_raw_cars_comments_5000.csv
数据处理：
- 随机抽取1000条数据
- 按4个类别（正面/负面/中性/无关）平均分配
- 划分训练集(800条)和验证集(200条)

输出位置：data/experiments/v1_1000samples/
├── train.csv          # 800条训练数据
├── val.csv           # 200条验证数据
└── data_stats.json   # 数据分布统计
```

#### 2. 训练阶段
```
模型配置：
- 预训练模型：BERT-base-chinese
- batch_size: 16
- learning_rate: 2e-5
- epochs: 5
- 特征：仅使用文本特征，不使用社交特征

输出位置：src/saved_models/
├── checkpoints/v1/
│   ├── best_model_1000samples.pth    # 最佳性能模型
│   └── latest_checkpoint.pth         # 最新检查点
├── logs/v1/
│   ├── training_log.txt             # 训练过程日志
│   └── experiment_record.md         # 实验记录
└── predictions/v1/
    ├── confusion_matrix.png         # 混淆矩阵
    └── error_analysis.txt           # 错误分析
```

#### 3. 分析阶段
- 模型性能评估
- 错误案例分析
- 各类别准确率分析
- 问题识别
- 改进方向规划

### 第二次训练 (3000条数据)

#### 1. 数据准备阶段
```
输入数据：
- 保留第一次的1000条数据
- 新增2000条数据（基于第一次错误分析结果）

数据处理重点：
- 增加第一次分类效果差的类别数据
- 补充特定模式的数据（如转折句）
- 确保数据质量和类别平衡

输出位置：data/experiments/v2_3000samples/
```

#### 2. 训练阶段
```
训练策略：
- 基于第一次训练的最佳模型继续训练
- 优化方向：
  * 学习率调整
  * 模型结构优化
  * 损失函数改进

输出位置：src/saved_models/checkpoints/v2/
```

#### 3. 分析阶段
- 与第一次训练结果对比
- 新的错误模式分析
- 改进效果评估
- 下一步优化方向确定

### 第三次训练 (6000条数据)

#### 1. 数据准备阶段
```
输入数据：
- 保留前3000条数据
- 新增3000条数据

数据处理重点：
- 基于前两次实验经验
- 重点关注难例
- 确保数据全面性

输出位置：data/experiments/v3_6000samples/
```

#### 2. 训练阶段
```
训练策略：
- 使用第二次训练的最佳模型继续训练
- 综合优化：
  * 最佳模型结构
  * 最优训练参数
  * 可能的集成策略

输出位置：src/saved_models/checkpoints/v3/
```

#### 3. 最终分析
- 三次训练的性能对比
- 模型能力提升分析
- 各类别的识别能力评估
- 最终模型的优缺点总结

## 实验记录规范

每次实验都会生成完整的记录：

### 1. 数据统计
- 样本数量和分布
- 数据特点分析
- 数据质量评估

### 2. 训练过程
- 损失变化曲线
- 准确率变化曲线
- 关键训练参数记录

### 3. 评估结果
- 准确率、精确率、召回率、F1分数
- 混淆矩阵分析
- 典型错误案例收集

### 4. 改进分析
- 存在问题总结
- 改进方向建议
- 下一步实验计划

## 目录结构

```
car_sentiment_analysis/
├── data/
│   ├── raw/                          # 原始数据
│   └── experiments/                  # 实验数据
│       ├── v1_1000samples/
│       ├── v2_3000samples/
│       └── v3_6000samples/
│
├── src/
│   ├── data_process/                 # 数据处理代码
│   ├── model_process/                # 模型处理代码
│   └── saved_models/                 # 模型保存
│       ├── checkpoints/              # 模型检查点
│       │       ├──v1
│       │       ├──v2
│       │       └──v3
│       ├── logs/    
│       │       ├──v1
│       │       ├──v2
│       │       └──v3                 # 训练日志
│       └── predictions/ 
│       │       ├──v1
│       │       ├──v2
│       └──     └──v3             # 预测结果
│
└── config/                           # 配置文件
```

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- transformers
- pandas
- scikit-learn
- matplotlib
- seaborn

## 使用说明


1. 模型训练：
```bash
python src/model_process/model_train.py     # 模型训练
```

2. 模型评估：
```bash
python src/model_process/model_evaluate.py   # 模型评估
```





汽车评论情感分析系统
这是一个基于BERT的汽车评论情感分析系统，使用主动学习策略提高标注效率。系统可以从数据库中读取评论数据，通过主动学习选择最有价值的样本进行人工标注，然后训练模型预测未标注数据的情感倾向，最后将结果回写到数据库。

一：项目结构
car_sentiment_analysis/
├── config/                     # 配置文件目录
│   ├── db_config.py            # 数据库连接配置
│   └── model_param_config.py   # 模型训练参数配置
├── src/                        # 源代码目录
│   ├── data_loader.py          # 从数据库加载评论数据
│   ├── model_train.py          # 模型训练模块
│   ├── model_evaluate.py       # 模型评估
│   ├── utils.py                #工具类  描述清楚工具
│   └── db_writer.py            # 预测结果回写模块
├── data/# 数据目录
│   ├── raw
│   │   ├── raw_comments.csv        # 原始评论数据
│   ├── processed
│   │   ├── labeled_data.csv        # 已标注数据
│   └── active_learning_samples.csv # 主动学习选择的待标注样本
├── model_output/               # 模型输出目录
│   ├── best_model/             # 最佳模型
│   └── final_model/            # 最终模型
├── .env                        # 环境变量配置文件（需自行创建）
├── README.md                   # 项目说明文档
└── requirements.txt            # 项目依赖

二：环境要求
Python 3.10
CUDA 12.3（用于GPU加速，可选）
RTX 3050或更高性能的GPU（可选）

三：使用说明
1. 安装依赖：`pip install -r requirements.txt`

四：配置数据库连接
在项目根目录创建.env文件，填入数据库连接信息：
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=car_sentiment
DB_TABLE=car_comments
TEXT_COLUMN=comment_text
ID_COLUMN=id
SENTIMENT_COLUMN=sentiment
`确保数据库中存在相应的表和字段


五：使用流程

2. 标注

- 0：标注为负面评论（吐槽、投诉、不满意）
- 1：标注为正面评论（称赞、满意、推荐）
- 2：标注为中性评论（客观描述、事实陈述）
- 3：标注为无关评论（与汽车情感无关的内容）






七：模型训练详解
本项目使用BERT（Bidirectional Encoder Representations from Transformers）模型进行情感分析。BERT是一种预训练语言模型，通过在大规模语料上预训练，可以捕捉文本的深层语义信息。

训练过程：
1. 数据准备：
   - 读取已标注数据
   - 划分训练集和验证集（默认80%/20%）
   - 使用BertTokenizer处理文本

2. 模型初始化：
   - 加载预训练的bert-base-chinese模型
   - 添加分类头（全连接层）

3. 训练配置：
   - 学习率：2e-5（BERT推荐学习率）
   - 批次大小：16（适合RTX 3050显存）
   - 训练轮数：5
   - 优化器：AdamW（带权重衰减的Adam）
   - 学习率调度：线性预热后线性衰减

4. 训练循环：
   - 前向传播计算损失
   - 反向传播更新参数
   - 每个epoch评估一次验证集性能
   - 保存最佳模型（基于F1分数）


5. 模型评估：
   - 准确率（Accuracy）：正确预测的比例
   - 精确率（Precision）：预测为正例中真正例的比例
   - 召回率（Recall）：真正例中被正确预测的比例
   - F1分数：精确率和召回率的调和平均

八：性能优化
1. 混合精度训练（FP16）：
   - 减少显存占用
   - 加速训练过程

2. 梯度累积：
   - 允许使用更大的等效批次大小
   - 适合显存受限的环境

3. 批量数据库操作：
   - 使用SQL的CASE语句进行批量更新
   - 显著提高数据库写入效率

4. 连接池：
   - 复用数据库连接
   - 减少连接建立和销毁的开销

九：依赖包说明
- torch：PyTorch深度学习框架，用于构建和训练BERT模型
- transformers：Hugging Face的Transformers库，提供预训练的BERT模型
- pandas：数据处理和分析库，用于CSV文件操作
- numpy：科学计算库，用于数值计算
- scikit-learn：机器学习工具库，用于数据分割和评估指标计算
- sqlalchemy：SQL工具包和ORM框架，用于数据库交互
- pymysql：MySQL数据库连接驱动
- tqdm：进度条显示库，提供直观的进度反馈
- matplotlib：绘图库，用于可视化训练过程
- python-dotenv：环境变量管理库，用于加载配置

十：常见问题与解决方案
1. 显存不足：
   - 减小batch_size
   - 启用梯度累积
   - 减小max_length参数

2. 数据库连接失败：
   - 检查.env文件配置
   - 确认数据库服务是否运行
   - 验证用户权限

3. 模型精度低：
   - 增加标注数据量
   - 调整模型参数
   - 尝试不同的预训练模型


在当前的实现中：
active_learning_label.py 中使用了一个简化版的BERT模型来计算不确定性，选择最有价值的样本进行标注。
model_train.py 实现了完整的BERT模型训练，用于最终的情感分析任务。
db_writer.py 使用训练好的模型对未标注数据进行预测并回写到数据库。
十一：主动学习与自动标注模型的关系

本项目中涉及两个模型：


2. **自动标注模型**：
   - 在`model_train.py`中实现
   - 使用所有已标注数据训练的完整BERT模型
   - 配置更为复杂，训练更加充分
   - 用于最终的情感分析预测任务
   - 预测结果通过`db_writer.py`回写到数据库

两个模型使用相同的预训练基础（bert-base-chinese），但训练目标不同：
- 主动学习模型优化目标是识别不确定样本
- 自动标注模型优化目标是最大化预测准确性

这种设计使得整个系统能够高效地利用人工标注资源，同时保证最终预测的质量。