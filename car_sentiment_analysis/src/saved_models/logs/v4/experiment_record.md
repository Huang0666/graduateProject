# 实验记录：第四次训练（大规模数据30000条）

开始时间：2025-04-27 17:12:46

## 实验配置
```python
{
  "experiment": {
    "description": "第四次训练（大规模数据30000条）",
    "focus": "从已有最优模型继续训练，优化模型性能"
  },
  "model_name": "bert-base-chinese",
  "num_labels": 4,
  "max_length": 128,
  "fp16": true,
  "evaluation_strategy": "epoch",
  "save_strategy": "epoch",
  "load_best_model_at_end": true,
  "metric_for_best_model": "f1",
  "output_dir": "src/saved_models",
  "data": {
    "train_path": "../../data/experiments/v4_30000samples/train.csv",
    "val_path": "../../data/experiments/v4_30000samples/val.csv",
    "data_size": 30000,
    "previous_model_path": "../saved_models/checkpoints/v3/best_model.pth"
  },
  "training": {
    "batch_size": 64,
    "learning_rate": 5e-06,
    "epochs": 15,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1
  },
  "save": {
    "checkpoint_dir": "../saved_models/checkpoints/v4",
    "log_dir": "../saved_models/logs/v4",
    "prediction_dir": "../saved_models/predictions/v4"
  }
}
```

### Epoch 1
- 训练损失：0.9433
- 验证损失：0.8622
- 验证F1：0.6945
- 训练准确率：0.6545
- 验证准确率：0.6946
- ROC AUC：0.8704

### Epoch 2
- 训练损失：0.8990
- 验证损失：0.7817
- 验证F1：0.6809
- 训练准确率：0.6526
- 验证准确率：0.6840
- ROC AUC：0.8963

### Epoch 3
- 训练损失：0.8029
- 验证损失：0.6886
- 验证F1：0.7275
- 训练准确率：0.6792
- 验证准确率：0.7280
- ROC AUC：0.9159

### Epoch 4
- 训练损失：0.7244
- 验证损失：0.6159
- 验证F1：0.7526
- 训练准确率：0.7061
- 验证准确率：0.7537
- ROC AUC：0.9333

### Epoch 5
- 训练损失：0.6438
- 验证损失：0.5569
- 验证F1：0.7766
- 训练准确率：0.7406
- 验证准确率：0.7762
- ROC AUC：0.9441

### Epoch 6
- 训练损失：0.5788
- 验证损失：0.5088
- 验证F1：0.7983
- 训练准确率：0.7678
- 验证准确率：0.7979
- ROC AUC：0.9531

### Epoch 7
- 训练损失：0.5331
- 验证损失：0.4802
- 验证F1：0.8094
- 训练准确率：0.7887
- 验证准确率：0.8081
- ROC AUC：0.9575

### Epoch 8
- 训练损失：0.4917
- 验证损失：0.4534
- 验证F1：0.8192
- 训练准确率：0.8045
- 验证准确率：0.8185
- ROC AUC：0.9619

### Epoch 9
- 训练损失：0.4620
- 验证损失：0.4360
- 验证F1：0.8252
- 训练准确率：0.8142
- 验证准确率：0.8244
- ROC AUC：0.9648

### Epoch 10
- 训练损失：0.4382
- 验证损失：0.4260
- 验证F1：0.8322
- 训练准确率：0.8249
- 验证准确率：0.8313
- ROC AUC：0.9660

### Epoch 11
- 训练损失：0.4256
- 验证损失：0.4225
- 验证F1：0.8311
- 训练准确率：0.8264
- 验证准确率：0.8294
- ROC AUC：0.9664

### Epoch 12
- 训练损失：0.4204
- 验证损失：0.4146
- 验证F1：0.8360
- 训练准确率：0.8316
- 验证准确率：0.8350
- ROC AUC：0.9676

### Epoch 13
- 训练损失：0.4104
- 验证损失：0.4129
- 验证F1：0.8367
- 训练准确率：0.8326
- 验证准确率：0.8356
- ROC AUC：0.9679

### Epoch 14
- 训练损失：0.4026
- 验证损失：0.4124
- 验证F1：0.8360
- 训练准确率：0.8393
- 验证准确率：0.8348
- ROC AUC：0.9679

### Epoch 15
- 训练损失：0.4040
- 验证损失：0.4099
- 验证F1：0.8377
- 训练准确率：0.8369
- 验证准确率：0.8365
- ROC AUC：0.9683
