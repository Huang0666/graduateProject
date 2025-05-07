"""
模型配置模块：整合所有模型训练、评估和数据增强的配置
"""

import os
from typing import Dict, Any

# 获取项目根目录的绝对路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 基础配置
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

# ===================== V1配置（1000条数据）=====================
V1_CONFIG = BASE_CONFIG.copy()
V1_CONFIG.update({
    'data': {
        'train_path': os.path.join(PROJECT_ROOT, 'data/experiments/v1_1000samples/train.csv'),
        'val_path': os.path.join(PROJECT_ROOT, 'data/experiments/v1_1000samples/val.csv'),
        'data_size': 1000
    },
    'training': {
        'batch_size': 16,
        'learning_rate': 2e-5,
        'epochs': 15,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 2
    },
    'save': {
        'checkpoint_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/checkpoints/v1'),
        'log_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/logs/v1'),
        'prediction_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/predictions/v1')
    },
    'experiment': {
        'version': 'v1',
        'description': '第一次训练（1000条数据）',
        'focus': '建立基准性能'
    }
})

# ===================== V2配置（3000条数据）=====================
V2_CONFIG = BASE_CONFIG.copy()
V2_CONFIG.update({
    'data': {
        'train_path': os.path.join(PROJECT_ROOT, 'data/experiments/v2_3000samples/train.csv'),
        'val_path': os.path.join(PROJECT_ROOT, 'data/experiments/v2_3000samples/val.csv'),
        'data_size': 3000,
        'previous_model_path': os.path.join(PROJECT_ROOT, 'src/saved_models/checkpoints/v1/best_model.pth')
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-5,
        'epochs': 25,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 2
    },
    'save': {
        'checkpoint_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/checkpoints/v2'),
        'log_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/logs/v2'),
        'prediction_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/predictions/v2')
    },
    'experiment': {
        'version': 'v2',
        'description': '第二次训练（3000条数据）',
        'focus': '优化模型性能，处理第一次训练中发现的问题'
    }
})

# ===================== V3配置（6000条数据）=====================
V3_CONFIG = BASE_CONFIG.copy()
V3_CONFIG.update({
    'data': {
        'train_path': os.path.join(PROJECT_ROOT, 'data/experiments/v3_6000samples/train.csv'),
        'val_path': os.path.join(PROJECT_ROOT, 'data/experiments/v3_6000samples/val.csv'),
        'data_size': 6000,
        'previous_model_path': os.path.join(PROJECT_ROOT, 'src/saved_models/checkpoints/v2/best_model.pth')
    },
    'training': {
        'batch_size': 64,
        'learning_rate': 5e-6,
        'epochs': 40,
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 1
    },
    'save': {
        'checkpoint_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/checkpoints/v3'),
        'log_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/logs/v3'),
        'prediction_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/predictions/v3')
    },
    'experiment': {
        'version': 'v3',
        'description': '第三次训练（6000条数据）',
        'focus': '最终优化，提升模型整体性能'
    }
})

# ===================== V4配置（30000条数据）=====================
V4_CONFIG = BASE_CONFIG.copy()
V4_CONFIG.update({
    'data': {
        'train_path': os.path.join(PROJECT_ROOT, 'data/experiments/v4_30000samples/train.csv'),
        'val_path': os.path.join(PROJECT_ROOT, 'data/experiments/v4_30000samples/val.csv'),
        'data_size': 30000,
        'previous_model_path': os.path.join(PROJECT_ROOT, 'src/saved_models/checkpoints/v3/best_model.pth')
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
        'checkpoint_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/checkpoints/v4'),
        'log_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/logs/v4'),
        'prediction_dir': os.path.join(PROJECT_ROOT, 'src/saved_models/predictions/v4')
    },
    'experiment': {
        'version': 'v4',
        'description': '第四次训练（30000条数据）',
        'focus': '从已有最优模型继续训练，优化模型性能'
    }
})

# ===================== 数据增强配置 =====================
class AugmentationConfig:
    """数据增强配置类"""
    
    def __init__(self):
        self.total_augment_ratio = 5.0  # 生成5倍全新数据
        
        # 词向量模型配置
        self.word2vec_config = {
            'model_path': os.path.join(PROJECT_ROOT, 'models/Tencent_AILab_ChineseEmbedding.txt'),
        }
        
        # 各方法比例配置
        self.method_ratios = {
            'template': 0.6,         # 模板替换 60%
            'synonym_pretrain': 0.4  # 预训练模型 40%
        }
        
        # 同义词替换配置
        self.synonym_config = {
            'min_word_length': 2,  # 最小词长度
            'max_replacements': 3,  # 最大替换数量
            'technical_threshold': 0.8,  # 专业术语相似度阈值
            'normal_threshold': 0.6,  # 普通词相似度阈值
        }
        
        # 模板替换配置
        self.template_config = {
            'min_template_length': 10,  # 最小模板长度
            'max_template_length': 500,  # 最大模板长度
        }
        
        # 采样配置
        self.sampling_config = {
            'random_seed': 42,  # 随机种子
        }
        
        # 输出配置
        self.output_config = {
            'save_intermediate': False,  # 是否保存中间结果
            'log_level': 'INFO',        # 日志级别
            'save_format': 'csv'        # 保存格式
        }
    
    def get_method_count(self, total_count: int) -> Dict[str, int]:
        """计算每种方法需要处理的数据量"""
        augment_count = int(total_count * self.total_augment_ratio)
        return {
            'template': int(augment_count * self.method_ratios['template']),
            'synonym_pretrain': int(augment_count * self.method_ratios['synonym_pretrain'])
        }
    
    def validate(self) -> bool:
        """验证配置的有效性"""
        try:
            method_sum = sum(self.method_ratios.values())
            if abs(method_sum - 1.0) > 0.001:
                print(f"方法比例之和应为1.0，当前为：{method_sum}")
                return False
            
            if not (0 < self.synonym_config['technical_threshold'] <= 1):
                print("专业术语相似度阈值应在(0,1]范围内")
                return False
                
            if not (0 < self.synonym_config['normal_threshold'] <= 1):
                print("普通词相似度阈值应在(0,1]范围内")
                return False
            
            if self.template_config['min_template_length'] >= self.template_config['max_template_length']:
                print("模板最小长度应小于最大长度")
                return False
            
            return True
            
        except Exception as e:
            print(f"配置验证失败：{e}")
            return False

# ===================== 评估配置 =====================
class EvaluationConfig:
    """评估配置类"""
    
    def __init__(self):
        # BERT模型配置
        self.bert_config = {
            'model_name': 'bert-base-chinese',
            'max_length': 512,
            'batch_size': 32
        }
        
        # 评估阈值配置
        self.thresholds = {
            'semantic_preservation': 0.8,  # 语义保持率阈值
            'sentiment_consistency': 0.9,  # 情感一致性阈值
            'domain_relevance': 0.6       # 领域相关性阈值
        }
        
        # 领域关键词配置
        self.domain_keywords = {
            '动力系统': {
                '发动机', '变速箱', '马力', '扭矩', '涡轮增压', '排量', '油耗',
                '动力', '加速', '换挡', '性能', '节油', '省油', '油门'
            },
            '操控系统': {
                '转向', '制动', '悬挂', '底盘', '刹车', '操控', '方向盘',
                '轮胎', '轮毂', '行驶', '稳定性', '操纵', '转弯', '路感'
            },
            '外观设计': {
                '车身', '前脸', '尾灯', '车灯', '大灯', '轮廓', '线条',
                '外形', '颜色', '造型', '外观', '设计', '时尚', '运动'
            },
            '内饰配置': {
                '中控', '座椅', '内饰', '储物', '空间', '材质', '做工',
                '舒适度', '质感', '用料', '装配', '细节', '用品', '配置'
            },
            '智能系统': {
                '车机', '辅助驾驶', 'HUD', '导航', '车联网', '屏幕', '系统',
                '功能', '智能', '科技', '互联', '自动', '感应', '控制'
            }
        }
        
        # 评估输出配置
        self.output_config = {
            'save_details': True,          # 是否保存详细评估结果
            'print_summary': True,         # 是否打印评估摘要
            'save_visualization': True     # 是否保存可视化结果
        }
    
    def validate(self) -> bool:
        """验证配置是否有效"""
        # 验证阈值范围
        for name, threshold in self.thresholds.items():
            if not 0 <= threshold <= 1:
                print(f"无效的阈值配置：{name} = {threshold}")
                return False
        
        # 验证领域关键词
        if not self.domain_keywords:
            print("领域关键词配置为空")
            return False
        
        # 验证BERT配置
        if not self.bert_config['model_name']:
            print("BERT模型名称未配置")
            return False
        
        if not 0 < self.bert_config['max_length'] <= 512:
            print(f"无效的最大长度配置：{self.bert_config['max_length']}")
            return False
        
        if self.bert_config['batch_size'] <= 0:
            print(f"无效的批处理大小：{self.bert_config['batch_size']}")
            return False
        
        return True
    
    def get_all_keywords(self) -> set:
        """获取所有领域关键词"""
        all_keywords = set()
        for keywords in self.domain_keywords.values():
            all_keywords.update(keywords)
        return all_keywords 