import os

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from .base_config import BASE_CONFIG

# 第二次实验配置（3000条数据）
V2_CONFIG = BASE_CONFIG.copy()
V2_CONFIG.update({
    # 数据配置
    'data': {
        'train_path': os.path.join(PROJECT_ROOT, 'data/experiments/v2_3000samples/train.csv'),
        'val_path': os.path.join(PROJECT_ROOT, 'data/experiments/v2_3000samples/val.csv'),
        'data_size': 3000,
        'previous_model_path': os.path.join(PROJECT_ROOT, 'src/saved_models/checkpoints/v1/best_model.pth')  # 加载第一次训练的最佳模型
    },
    
    # 训练参数（优化后的参数）
    'training': {
        'batch_size': 32,  # 增大批次大小
        'learning_rate': 1e-5,  # 降低学习率
        'epochs': 25,  # 增加训练轮次
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 2
    },
    
    # 保存路径
    'save': {
        'checkpoint_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/checkpoints/v2'),
        'log_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/logs/v2'),
        'prediction_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/predictions/v2')
    },
    
    # 实验记录
    'experiment': {
        'version': 'v2',
        'description': '第二次训练（3000条数据）',
        'focus': '优化模型性能，处理第一次训练中发现的问题'
    }
}) 