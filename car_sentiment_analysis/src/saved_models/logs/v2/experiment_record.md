# 实验记录：第二次训练（3000条数据）

开始时间：2025-04-24 12:26:00

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
    "train_path": "D:\\graduation_project\\car_sentiment_analysis\\data/experiments/v2_3000samples/train.csv",
    "val_path": "D:\\graduation_project\\car_sentiment_analysis\\data/experiments/v2_3000samples/val.csv",
    "data_size": 3000,
    "previous_model_path": "D:\\graduation_project\\car_sentiment_analysis\\src/saved_models/checkpoints/v1/best_model.pth"
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 1e-05,
    "epochs": 25,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 2
  },
  "save": {
    "checkpoint_dir": "D:\\graduation_project\\car_sentiment_analysis\\src/saved_models/checkpoints/v2",
    "log_dir": "D:\\graduation_project\\car_sentiment_analysis\\src/saved_models/logs/v2",
    "prediction_dir": "D:\\graduation_project\\car_sentiment_analysis\\src/saved_models/predictions/v2"
  },
  "experiment": {
    "version": "v2",
    "description": "第二次训练（3000条数据）",
    "focus": "优化模型性能，处理第一次训练中发现的问题"
  }
}
```

### Epoch 1
- 训练损失：1.2599
- 验证损失：1.1899
- 验证F1：0.4977
- 训练准确率：0.4883
- 验证准确率：0.5067
- ROC AUC：0.7547

### Epoch 2
- 训练损失：1.2505
- 验证损失：1.1575
- 验证F1：0.4898
- 训练准确率：0.4879
- 验证准确率：0.5050
- ROC AUC：0.7626

### Epoch 3
- 训练损失：1.2083
- 验证损失：1.1319
- 验证F1：0.5010
- 训练准确率：0.4975
- 验证准确率：0.5200
- ROC AUC：0.7704

### Epoch 4
- 训练损失：1.1772
- 验证损失：1.0902
- 验证F1：0.5199
- 训练准确率：0.5121
- 验证准确率：0.5283
- ROC AUC：0.7822

### Epoch 5
- 训练损失：1.1391
- 验证损失：1.0776
- 验证F1：0.5169
- 训练准确率：0.5317
- 验证准确率：0.5317
- ROC AUC：0.7873

### Epoch 6
- 训练损失：1.0959
- 验证损失：1.0750
- 验证F1：0.5175
- 训练准确率：0.5454
- 验证准确率：0.5317
- ROC AUC：0.7906

### Epoch 7
- 训练损失：1.0764
- 验证损失：1.0596
- 验证F1：0.5271
- 训练准确率：0.5633
- 验证准确率：0.5433
- ROC AUC：0.7956

### Epoch 8
- 训练损失：1.0540
- 验证损失：1.0440
- 验证F1：0.5400
- 训练准确率：0.5737
- 验证准确率：0.5417
- ROC AUC：0.8001

### Epoch 9
- 训练损失：1.0321
- 验证损失：1.0458
- 验证F1：0.5379
- 训练准确率：0.5929
- 验证准确率：0.5400
- ROC AUC：0.8037

### Epoch 10
- 训练损失：1.0084
- 验证损失：1.0244
- 验证F1：0.5356
- 训练准确率：0.5967
- 验证准确率：0.5383
- ROC AUC：0.8053

### Epoch 11
- 训练损失：0.9747
- 验证损失：1.0335
- 验证F1：0.5351
- 训练准确率：0.6150
- 验证准确率：0.5517
- ROC AUC：0.8065

### Epoch 12
- 训练损失：0.9537
- 验证损失：1.0135
- 验证F1：0.5433
- 训练准确率：0.6271
- 验证准确率：0.5517
- ROC AUC：0.8102

### Epoch 13
- 训练损失：0.9390
- 验证损失：1.0100
- 验证F1：0.5422
- 训练准确率：0.6338
- 验证准确率：0.5500
- ROC AUC：0.8120

### Epoch 14
- 训练损失：0.9259
- 验证损失：1.0239
- 验证F1：0.5529
- 训练准确率：0.6479
- 验证准确率：0.5600
- ROC AUC：0.8120

### Epoch 15
- 训练损失：0.9099
- 验证损失：1.0146
- 验证F1：0.5525
- 训练准确率：0.6425
- 验证准确率：0.5567
- ROC AUC：0.8132

### Epoch 16
- 训练损失：0.9040
- 验证损失：1.0239
- 验证F1：0.5540
- 训练准确率：0.6558
- 验证准确率：0.5650
- ROC AUC：0.8128

### Epoch 17
- 训练损失：0.8818
- 验证损失：1.0063
- 验证F1：0.5565
- 训练准确率：0.6529
- 验证准确率：0.5633
- ROC AUC：0.8158

### Epoch 18
- 训练损失：0.8552
- 验证损失：1.0069
- 验证F1：0.5569
- 训练准确率：0.6654
- 验证准确率：0.5633
- ROC AUC：0.8169

### Epoch 19
- 训练损失：0.8545
- 验证损失：1.0149
- 验证F1：0.5711
- 训练准确率：0.6742
- 验证准确率：0.5817
- ROC AUC：0.8166

### Epoch 20
- 训练损失：0.8362
- 验证损失：1.0031
- 验证F1：0.5665
- 训练准确率：0.6829
- 验证准确率：0.5700
- ROC AUC：0.8185

### Epoch 21
- 训练损失：0.8311
- 验证损失：1.0144
- 验证F1：0.5663
- 训练准确率：0.6817
- 验证准确率：0.5750
- ROC AUC：0.8188

### Epoch 22
- 训练损失：0.8330
- 验证损失：1.0106
- 验证F1：0.5723
- 训练准确率：0.6804
- 验证准确率：0.5783
- ROC AUC：0.8188

### Epoch 23
- 训练损失：0.8218
- 验证损失：1.0063
- 验证F1：0.5722
- 训练准确率：0.6900
- 验证准确率：0.5783
- ROC AUC：0.8197

### Epoch 24
- 训练损失：0.8180
- 验证损失：1.0051
- 验证F1：0.5765
- 训练准确率：0.6892
- 验证准确率：0.5817
- ROC AUC：0.8201

### Epoch 25
- 训练损失：0.8222
- 验证损失：1.0061
- 验证F1：0.5760
- 训练准确率：0.6925
- 验证准确率：0.5817
- ROC AUC：0.8200
