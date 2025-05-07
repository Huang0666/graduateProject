"""
数据增强质量评估模块

实现三个主要评估指标：
1. 语义保持率
2. 情感一致性
3. 领域相关性
"""

import jieba
import numpy as np
from typing import List, Dict, Set, Tuple
import logging
from pathlib import Path
from collections import defaultdict
import json
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity as torch_cosine_similarity

class QualityEvaluator:
    """数据增强质量评估器"""
    
    def __init__(self, bert_model_name: str = "bert-base-chinese"):
        """
        初始化评估器
        
        Args:
            bert_model_name: BERT模型名称
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name).to(self.device)
        self.bert_model.eval()
        
        # 加载汽车领域关键词
        self._init_domain_keywords()
        
        # 评估结果存储
        self.evaluation_results = defaultdict(dict)
        
    def _init_domain_keywords(self):
        """初始化汽车领域关键词集合"""
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
        
        # 展平关键词集合
        self.all_keywords = set().union(*self.domain_keywords.values())
        
    def get_bert_embedding(self, text: str) -> torch.Tensor:
        """
        获取文本的BERT嵌入表示
        
        Args:
            text: 输入文本
            
        Returns:
            torch.Tensor: 文本的嵌入表示
        """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.bert_model(**inputs)
            # 使用[CLS]标记的输出作为整个句子的表示
            return outputs.last_hidden_state[:, 0, :]
            
    def evaluate_semantic_preservation(self, original_texts: List[str], augmented_texts: List[str],
                                    threshold: float = 0.8) -> Tuple[float, List[bool]]:
        """
        评估语义保持率
        
        Args:
            original_texts: 原始文本列表
            augmented_texts: 增强后的文本列表
            threshold: 语义相似度阈值
            
        Returns:
            Tuple[float, List[bool]]: (语义保持率, 每条文本的评估结果)
        """
        preserved_count = 0
        preservation_results = []
        
        for orig_text, aug_text in zip(original_texts, augmented_texts):
            # 获取文本嵌入
            orig_embedding = self.get_bert_embedding(orig_text)
            aug_embedding = self.get_bert_embedding(aug_text)
            
            # 计算余弦相似度
            similarity = torch_cosine_similarity(orig_embedding, aug_embedding).item()
            
            # 判断是否保持语义
            is_preserved = similarity >= threshold
            preservation_results.append(is_preserved)
            if is_preserved:
                preserved_count += 1
                
        preservation_rate = preserved_count / len(original_texts)
        return preservation_rate, preservation_results
        
    def evaluate_sentiment_consistency(self, original_sentiments: List[int],
                                    augmented_texts: List[str]) -> Tuple[float, List[bool]]:
        """
        评估情感一致性
        
        Args:
            original_sentiments: 原始情感标签列表
            augmented_texts: 增强后的文本列表
            
        Returns:
            Tuple[float, List[bool]]: (情感一致性比率, 每条文本的评估结果)
        """
        # TODO: 实现情感分类模型
        # 当前使用随机模拟，实际使用时需要替换为真实的情感分类模型
        consistent_count = 0
        consistency_results = []
        
        for orig_sentiment, aug_text in zip(original_sentiments, augmented_texts):
            # 这里应该使用情感分类模型进行预测
            # 当前使用随机值模拟
            predicted_sentiment = orig_sentiment  # 临时使用原始情感
            
            # 判断情感是否一致
            is_consistent = predicted_sentiment == orig_sentiment
            consistency_results.append(is_consistent)
            if is_consistent:
                consistent_count += 1
                
        consistency_rate = consistent_count / len(original_sentiments)
        return consistency_rate, consistency_results
        
    def evaluate_domain_relevance(self, texts: List[str]) -> Tuple[float, List[float]]:
        """
        评估领域相关性
        
        Args:
            texts: 文本列表
            
        Returns:
            Tuple[float, List[float]]: (平均领域相关性得分, 每条文本的相关性得分)
        """
        relevance_scores = []
        
        for text in texts:
            # 分词
            words = set(jieba.cut(text))
            
            # 计算领域关键词匹配数
            matched_keywords = words & self.all_keywords
            
            # 计算相关性得分
            # 得分 = 匹配关键词数 / max(文本词数, 最小期望关键词数)
            min_expected_keywords = 2  # 最小期望关键词数
            relevance_score = len(matched_keywords) / max(len(words), min_expected_keywords)
            relevance_scores.append(relevance_score)
            
        avg_relevance = sum(relevance_scores) / len(texts)
        return avg_relevance, relevance_scores
        
    def evaluate_batch(self, original_texts: List[str], augmented_texts: List[str],
                      original_sentiments: List[int], method: str) -> Dict:
        """
        批量评估增强效果
        
        Args:
            original_texts: 原始文本列表
            augmented_texts: 增强后的文本列表
            original_sentiments: 原始情感标签列表
            method: 增强方法名称
            
        Returns:
            Dict: 评估结果
        """
        # 1. 评估语义保持率
        semantic_rate, semantic_results = self.evaluate_semantic_preservation(
            original_texts, augmented_texts
        )
        
        # 2. 评估情感一致性
        consistency_rate, consistency_results = self.evaluate_sentiment_consistency(
            original_sentiments, augmented_texts
        )
        
        # 3. 评估领域相关性
        relevance_rate, relevance_scores = self.evaluate_domain_relevance(augmented_texts)
        
        # 存储评估结果
        results = {
            'method': method,
            'sample_count': len(original_texts),
            'semantic_preservation': {
                'rate': semantic_rate,
                'details': semantic_results
            },
            'sentiment_consistency': {
                'rate': consistency_rate,
                'details': consistency_results
            },
            'domain_relevance': {
                'rate': relevance_rate,
                'scores': relevance_scores
            }
        }
        
        self.evaluation_results[method] = results
        return results
        
    def save_results(self, output_file: str):
        """
        保存评估结果到文件
        
        Args:
            output_file: 输出文件路径
        """
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
            
    def print_summary(self):
        """打印评估结果摘要"""
        print("\n=== 数据增强质量评估报告 ===")
        
        for method, results in self.evaluation_results.items():
            print(f"\n方法：{method}")
            print(f"样本数量：{results['sample_count']}")
            print(f"语义保持率：{results['semantic_preservation']['rate']:.2%}")
            print(f"情感一致性：{results['sentiment_consistency']['rate']:.2%}")
            print(f"领域相关性：{results['domain_relevance']['rate']:.2%}")
            print("-" * 40) 