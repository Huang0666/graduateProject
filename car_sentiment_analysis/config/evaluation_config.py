"""评估配置模块"""

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
        """
        验证配置是否有效
        
        Returns:
            bool: 配置是否有效
        """
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
        """
        获取所有领域关键词
        
        Returns:
            set: 所有关键词集合
        """
        all_keywords = set()
        for keywords in self.domain_keywords.values():
            all_keywords.update(keywords)
        return all_keywords 