import os

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from .base_config import BASE_CONFIG

# 第一次实验配置（1000条数据）
V1_CONFIG = BASE_CONFIG.copy()
V1_CONFIG.update({
    # 数据配置
    'data': {
        'train_path': os.path.join(PROJECT_ROOT, 'data/experiments/v1_1000samples/train.csv'),
        'val_path': os.path.join(PROJECT_ROOT, 'data/experiments/v1_1000samples/val.csv'),
        'data_size': 1000
    },
    
    # 训练参数
    'training': {
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 15,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 2
    },
    
    # 保存路径
    'save': {
        'checkpoint_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/checkpoints/v1'),
        'log_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/logs/v1'),
        'prediction_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/predictions/v1')
    },
    
    # 实验记录
    'experiment': {
        'version': 'v1',
        'description': '第一次训练（1000条数据）',
        'focus': '建立基准性能'
    }
}) 