# 实验记录：第一次训练（1000条数据）

开始时间：2025-04-24 12:05:28

## 实验配置
```python
{
  "model_name": "bert-base-chinese",
  "max_length": 128,
  "num_labels": 4,
  "fp16": true,
  "evaluation_strategy": "epoch",
  "save_strategy": "epoch",
  "load_best_model_at_end": true,
  "metric_for_best_model": "f1",
  "output_dir": "src/saved_models",
  "data": {
    "train_path": "D:\\graduation_project\\car_sentiment_analysis\\data/experiments/v1_1000samples/train.csv",
    "val_path": "D:\\graduation_project\\car_sentiment_analysis\\data/experiments/v1_1000samples/val.csv",
    "data_size": 1000
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 2e-05,
    "epochs": 15,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 2
  },
  "save": {
    "checkpoint_dir": "D:\\graduation_project\\car_sentiment_analysis\\src/saved_models/checkpoints/v1",
    "log_dir": "D:\\graduation_project\\car_sentiment_analysis\\src/saved_models/logs/v1",
    "prediction_dir": "D:\\graduation_project\\car_sentiment_analysis\\src/saved_models/predictions/v1"
  },
  "experiment": {
    "version": "v1",
    "description": "第一次训练（1000条数据）",
    "focus": "建立基准性能"
  }
}
```

### Epoch 1
- 训练损失：1.7959
- 验证损失：1.5142
- 验证F1：0.1330
- 训练准确率：0.2275
- 验证准确率：0.1900
- ROC AUC：0.4973

### Epoch 2
- 训练损失：1.6939
- 验证损失：1.4425
- 验证F1：0.2342
- 训练准确率：0.2662
- 验证准确率：0.2400
- ROC AUC：0.5392

### Epoch 3
- 训练损失：1.5460
- 验证损失：1.3924
- 验证F1：0.3404
- 训练准确率：0.2875
- 验证准确率：0.3550
- ROC AUC：0.5972

### Epoch 4
- 训练损失：1.4599
- 验证损失：1.3401
- 验证F1：0.3730
- 训练准确率：0.3225
- 验证准确率：0.3950
- ROC AUC：0.6486

### Epoch 5
- 训练损失：1.3518
- 验证损失：1.2853
- 验证F1：0.4578
- 训练准确率：0.4375
- 验证准确率：0.4650
- ROC AUC：0.6827

### Epoch 6
- 训练损失：1.2936
- 验证损失：1.2493
- 验证F1：0.4486
- 训练准确率：0.4325
- 验证准确率：0.4600
- ROC AUC：0.6958

### Epoch 7
- 训练损失：1.2545
- 验证损失：1.2027
- 验证F1：0.4443
- 训练准确率：0.4650
- 验证准确率：0.4650
- ROC AUC：0.7278

### Epoch 8
- 训练损失：1.2207
- 验证损失：1.1718
- 验证F1：0.4843
- 训练准确率：0.4888
- 验证准确率：0.4950
- ROC AUC：0.7485

### Epoch 9
- 训练损失：1.1676
- 验证损失：1.1374
- 验证F1：0.4788
- 训练准确率：0.5288
- 验证准确率：0.4950
- ROC AUC：0.7662

### Epoch 10
- 训练损失：1.1251
- 验证损失：1.1264
- 验证F1：0.4996
- 训练准确率：0.5775
- 验证准确率：0.5100
- ROC AUC：0.7765

### Epoch 11
- 训练损失：1.1101
- 验证损失：1.0954
- 验证F1：0.5401
- 训练准确率：0.5525
- 验证准确率：0.5450
- ROC AUC：0.7870

### Epoch 12
- 训练损失：1.0875
- 验证损失：1.0926
- 验证F1：0.5129
- 训练准确率：0.5825
- 验证准确率：0.5200
- ROC AUC：0.7922

### Epoch 13
- 训练损失：1.0738
- 验证损失：1.0783
- 验证F1：0.5228
- 训练准确率：0.5975
- 验证准确率：0.5300
- ROC AUC：0.7996

### Epoch 14
- 训练损失：1.0521
- 验证损失：1.0733
- 验证F1：0.5368
- 训练准确率：0.6150
- 验证准确率：0.5450
- ROC AUC：0.7991

### Epoch 15
- 训练损失：1.0482
- 验证损失：1.0762
- 验证F1：0.5180
- 训练准确率：0.5988
- 验证准确率：0.5250
- ROC AUC：0.7993
