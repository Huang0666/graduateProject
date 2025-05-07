# 情感分析模块实现文档

## 1. 模块概述

`sentiment_analysis.py`是情感分析系统的实际应用模块，负责对数据库中的评论进行批量情感分析。该模块集成了训练好的模型和优化后的阈值策略，实现了高效的批量处理功能。

## 2. 核心功能

### 2.1 阈值策略实现
```python
def load_threshold_config(config_path='threshold_config.json'):
    """加载阈值配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            threshold = config.get('best_threshold', 0.5)
            logging.info(f"已加载阈值配置：{threshold}")
            return threshold
    except Exception as e:
        logging.warning(f"加载阈值配置失败：{e}，使用默认阈值0.5")
        return 0.5
```

阈值应用策略：
```python
# 应用阈值策略
positive_probs = probs[:, 1]  # 正向概率
batch_preds = []

for prob in positive_probs:
    if prob > self.threshold:
        batch_preds.append(1)  # 正向
    else:
        # 在其他类别中选择最高概率
        other_probs = probs.clone()
        other_probs[:, 1] = 0
        pred = other_probs.argmax(dim=1).item()
        batch_preds.append(pred)
```

### 2.2 批量处理机制

1. **数据加载**：
   - 使用自定义的`CommentDataset`类
   - 支持社交特征（点赞数和回复数）
   - 实现数据验证和异常处理

2. **批处理流程**：
   - 使用DataLoader进行批量处理
   - 支持GPU加速
   - 实现进度显示和日志记录

### 2.3 错误处理和重试机制

1. **重试策略**：
   - 最大重试次数可配置
   - 失败记录保存到日志
   - 重试间隔为1秒

2. **错误记录**：
   - 详细的错误信息记录
   - 失败ID列表保存
   - 支持后续手动处理

## 3. 配置说明

### 3.1 模型配置
```python
MODEL_VERSIONS = {
    'v3': {
        'path': 'src/saved_models/checkpoints/v3/best_model.pth',
        'description': '6000条数据训练的四分类模型',
        'config': REQUIRED_CONFIG
    },
    'v4': {
        'path': 'src/saved_models/checkpoints/v4/best_model.pth',
        'description': '30000条数据训练的四分类模型',
        'config': REQUIRED_CONFIG
    }
}
```

### 3.2 阈值配置
```json
{
    "best_threshold": 0.4,
    "description": "基于V4模型优化后的最佳阈值",
    "model_version": "v4",
    "optimization_date": "2024-03"
}
```

## 4. 使用说明

### 4.1 命令行参数
```bash
python sentiment_analysis.py --model_version v4 --batch_size 32 --max_retries 3
```

### 4.2 环境要求
- Python 3.6+
- PyTorch
- transformers
- mysql-connector-python
- python-dotenv
- tqdm

## 5. 注意事项

1. 确保`threshold_config.json`位于正确路径
2. 数据库连接信息需要在`.env`文件中配置
3. 建议使用GPU进行处理以提高效率
4. 处理失败的记录会保存在日志文件中 