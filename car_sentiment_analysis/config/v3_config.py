import os

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from .base_config import BASE_CONFIG

# 第三次实验配置（6000条数据）
V3_CONFIG = BASE_CONFIG.copy()
V3_CONFIG.update({
    # 数据配置
    'data': {
        'train_path': os.path.join(PROJECT_ROOT, 'data/experiments/v3_6000samples/train.csv'),
        'val_path': os.path.join(PROJECT_ROOT, 'data/experiments/v3_6000samples/val.csv'),
        'data_size': 6000,
        'previous_model_path': os.path.join(PROJECT_ROOT, 'src/saved_models/checkpoints/v2/best_model.pth')  # 加载第二次训练的最佳模型
    },
    
    # 训练参数（最终优化）
    'training': {
        'batch_size': 64,  # 进一步增大批次
        'learning_rate': 5e-6,  # 进一步降低学习率
        'epochs':40,  # 增加训练轮次
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 1  # 根据显存调整
    },
    
    # 保存路径
    'save': {
        'checkpoint_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/checkpoints/v3'),
        'log_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/logs/v3'),
        'prediction_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/predictions/v3')
    },
    
    # 实验记录
    'experiment': {
        'version': 'v3',
        'description': '第三次训练（6000条数据）',
        'focus': '最终优化，提升模型整体性能'
    }
}) 