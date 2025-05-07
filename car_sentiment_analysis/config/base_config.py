# 基础配置文件，包含所有实验共享的基本参数
BASE_CONFIG = {
    # 模型基础配置
    'model_name': 'bert-base-chinese',
    'max_length': 128,
    'num_labels': 4,  # 四分类任务：负面、正面、中性、无关
    
    # 训练基础配置
    'fp16': True,                       # 混合精度训练
    'evaluation_strategy': 'epoch',     # 每个epoch评估
    'save_strategy': 'epoch',          # 每个epoch保存
    'load_best_model_at_end': True,    # 训练结束后加载最佳模型
    'metric_for_best_model': 'f1',     # 用f1分数选择最佳模型
    
    # 输出目录配置
    'output_dir': 'src/saved_models',
} 