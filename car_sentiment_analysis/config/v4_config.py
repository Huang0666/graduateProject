"""
第四版训练配置（30000条数据）
"""

V4_CONFIG = {
    'experiment': {
        'description': '第四次训练（大规模数据30000条）',
        'focus': '从已有最优模型继续训练，优化模型性能'
    },
    
    'model_name': 'bert-base-chinese',
    'num_labels': 4,
    'max_length': 128,
    'fp16': True,
    'evaluation_strategy': 'epoch',
    'save_strategy': 'epoch',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'f1',
    'output_dir': 'src/saved_models',
    
    'data': {
        'train_path': '../../data/experiments/v4_30000samples/train.csv',
        'val_path': '../../data/experiments/v4_30000samples/val.csv',
        'data_size': 30000,
        'previous_model_path': '../saved_models/checkpoints/v3/best_model.pth'
    },
    
    'training': {
        'batch_size': 64,
        'learning_rate': 5e-6,
        'epochs': 15,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 1
    },
    
    'save': {
        'checkpoint_dir': '../saved_models/checkpoints/v4',
        'log_dir': '../saved_models/logs/v4',
        'prediction_dir': '../saved_models/predictions/v4'
    }
} 