"""
汽车评论数据增强模块

实现两种数据增强策略：
1. 模板替换增强（60%）
2. 基于词向量的相似词替换（40%）

改进：
- 扩展技术术语库
- 增强质量控制
- 优化同义词替换策略
- 添加数据清洗功能
"""

import random
import jieba
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Set, Tuple, Any
import argparse
import os
import json
from pathlib import Path
import logging
import numpy as np
from gensim.models import KeyedVectors
import re

# 添加项目根目录到系统路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.augmentation_config import AugmentationConfig
from src.data_process.quality_evaluator import QualityEvaluator

# 全局变量
word2vec_model = None

def load_word2vec_model(model_path: str):
    """加载词向量模型"""
    global word2vec_model
    if word2vec_model is None:
        logging.info(f"加载词向量模型：{model_path}")
        word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    return word2vec_model

class AutoTemplateGenerator:
    """自动模板生成器"""
    
    def __init__(self, config: AugmentationConfig):
        """
        初始化模板生成器
        
        Args:
            config: 增强配置对象
        """
        self.config = config
        self.templates = {
            '0': [],  # 负面评论模板
            '1': [],  # 正面评论模板
            '2': [],  # 中性评论模板
            '3': []   # 无关评论模板
        }
        
    def extract_templates(self, df: pd.DataFrame) -> None:
        """
        从数据集中提取模板
        
        Args:
            df: 原始数据DataFrame
        """
        # 获取所有车型名称
        car_names = df['car_name'].unique()
        
        # 按情感标签分组处理
        for sentiment in ['0', '1', '2', '3']:  # 0负面, 1正面, 2中性, 3无关
            sentiment_df = df[df['sentiment_analysis_results'] == int(sentiment)]
            
            for _, row in sentiment_df.iterrows():
                content = row['content']
                # 提取模板：替换车名为占位符
                template = content
                for car_name in car_names:
                    if car_name in content:
                        template = template.replace(car_name, '{car_name}')
                
                # 检查模板是否符合要求
                if self._is_valid_template(template):
                    self.templates[sentiment].append(template)
            
            # 去重
            self.templates[sentiment] = list(set(self.templates[sentiment]))
            
        # 打印统计信息
        logging.info("模板统计信息：")
        sentiment_names = {
            '0': '负面',
            '1': '正面',
            '2': '中性',
            '3': '无关'
        }
        for sentiment, templates in self.templates.items():
            logging.info(f"情感 {sentiment_names[sentiment]}: {len(templates)} 个模板")
            
    def _is_valid_template(self, template: str) -> bool:
        """
        检查模板是否有效
        
        Args:
            template: 待检查的模板
            
        Returns:
            bool: 模板是否有效
        """
        # 检查长度
        min_length = self.config.template_config.get('min_template_length', 10)
        max_length = self.config.template_config.get('max_template_length', 200)
        
        if not (min_length <= len(template) <= max_length):
            return False
            
        # 检查是否包含占位符
        if '{car_name}' not in template:
            return False
            
        return True
        
    def generate(self, text: str) -> str:
        """
        生成新的评论
        
        Args:
            text: 原始文本
            
        Returns:
            str: 生成的新文本
        """
        # 从原始文本中提取车型名称
        car_name = None
        for word in jieba.cut(text):
            if "车" in word or "汽车" in word:
                car_name = word
                break
                
        if not car_name:
            return text
            
        # 获取情感倾向
        sentiment = '1'  # 默认使用正面情感模板
        if "不" in text or "差" in text or "烂" in text:
            sentiment = '0'
            
        # 获取对应情感的模板
        available_templates = self.templates.get(sentiment, [])
        if not available_templates:
            return text
            
        # 随机选择一个模板
        template = random.choice(available_templates)
        
        # 替换占位符
        new_text = template.format(car_name=car_name)
        
        # 避免生成重复的评论
        if new_text == text:
            return self.generate(text)
            
        return new_text

class SynonymAugmenter:
    """同义词增强器"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self._init_technical_terms()
        
    def _init_technical_terms(self):
        """初始化技术术语集"""
        self.technical_terms = {
            '动力系统': ['发动机', '变速箱', '马力', '扭矩', '涡轮增压', '排量', '油耗', 
                     '动力', '加速', '换挡', 'CVT', 'DCT', '自动挡', '手动挡'],
            '操控系统': ['转向', '制动', '悬挂', '底盘', '刹车', '操控', '方向盘', 
                     '轮胎', '轮毂', '行驶', 'ESP', 'ABS', '转向助力'],
            '外观设计': ['车身', '前脸', '尾灯', '车灯', '大灯', '轮廓', '线条', 
                     '外形', '颜色', '造型', '格栅', '后视镜', '车门'],
            '内饰配置': ['中控', '座椅', '内饰', '储物', '空间', '材质', '做工', 
                     '舒适度', '质感', '仪表盘', '音响', '空调', '扶手']
        }
        # 展平技术术语列表
        self.all_technical_terms = set(
            term for terms in self.technical_terms.values() 
            for term in terms
        )

    def augment_with_word2vec(self, text: str) -> str:
        """
        使用词向量进行相似词替换增强
        
        Args:
            text: 输入文本
            
        Returns:
            str: 增强后的文本
        """
        if word2vec_model is None:
            model_path = self.config.synonym_config.get('word2vec_path')
            if not model_path:
                return text
            load_word2vec_model(model_path)
            if word2vec_model is None:
                return text
                
        # 分词
        words = list(jieba.cut(text))
        new_words = words.copy()
        
        # 获取替换概率和相似度阈值
        replace_prob = self.config.synonym_config.get('word2vec_replace_prob', 0.3)
        similarity_threshold = self.config.synonym_config.get('similarity_threshold', 0.7)
        
        # 遍历词语进行替换
        for i, word in enumerate(words):
            # 跳过技术术语
            if word in self.all_technical_terms:
                continue
                
            # 按概率决定是否替换
            if random.random() > replace_prob:
                continue
                
            try:
                # 获取相似词
                similar_words = word2vec_model.most_similar(word, topn=5)
                # 过滤掉相似度低于阈值的词
                similar_words = [w for w, sim in similar_words if sim >= similarity_threshold]
                
                if similar_words:
                    # 随机选择一个相似词替换
                    new_word = random.choice(similar_words)
                    new_words[i] = new_word
            except Exception:
                # 出错时静默处理
                continue
                
        return ''.join(new_words)

class TextAugmenter:
    """文本增强主类：协调各个增强组件"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.template_generator = AutoTemplateGenerator(config)
        self.synonym_augmenter = SynonymAugmenter(config)
        
    def augment(self, df: pd.DataFrame) -> Dict:
        """
        执行数据增强
        
        Args:
            df: 原始数据DataFrame
        
        Returns:
            Dict: 增强结果，包含原始文本和增强文本的对应关系
        """
        logging.info("开始数据增强...")
        
        # 提取模板
        logging.info("提取评论模板...")
        self.template_generator.extract_templates(df)
        
        augmented_data = []
        augment_info = {
            'template': {'original': [], 'augmented': [], 'sentiments': []},
            'synonym_pretrain': {'original': [], 'augmented': [], 'sentiments': []}
        }
        
        # 获取每种方法需要处理的数据量
        method_counts = self.config.get_method_count(len(df))
        
        # 1. 模板替换增强
        if method_counts['template'] > 0:
            logging.info(f"执行模板替换增强，目标数量：{method_counts['template']}")
            template_df = df.sample(n=method_counts['template'], replace=True)
            
            for _, row in template_df.iterrows():
                original_text = row['content']
                augmented_text = self.template_generator.generate(original_text)
                
                new_row = row.copy()
                new_row['content'] = augmented_text
                new_row['aug_method'] = 'template_replace'
                augmented_data.append(new_row)
                
                # 记录用于评估的信息
                augment_info['template']['original'].append(original_text)
                augment_info['template']['augmented'].append(augmented_text)
                augment_info['template']['sentiments'].append(row['sentiment_analysis_results'])
        
        # 2. 词向量相似词替换
        if method_counts['synonym_pretrain'] > 0:
            logging.info(f"执行词向量相似词替换，目标数量：{method_counts['synonym_pretrain']}")
            pretrain_df = df.sample(n=method_counts['synonym_pretrain'], replace=True)
            
            for _, row in pretrain_df.iterrows():
                original_text = row['content']
                augmented_text = self.synonym_augmenter.augment_with_word2vec(original_text)
                
                new_row = row.copy()
                new_row['content'] = augmented_text
                new_row['aug_method'] = 'word2vec_similar'
                augmented_data.append(new_row)
                
                # 记录用于评估的信息
                augment_info['synonym_pretrain']['original'].append(original_text)
                augment_info['synonym_pretrain']['augmented'].append(augmented_text)
                augment_info['synonym_pretrain']['sentiments'].append(row['sentiment_analysis_results'])
        
        # 转换为DataFrame并重置索引
        augmented_df = pd.DataFrame(augmented_data)
        
        # 如果已存在id列，先删除
        if 'id' in augmented_df.columns:
            augmented_df.drop('id', axis=1, inplace=True)
            
        # 重置索引并添加新的id列
        augmented_df.reset_index(drop=True, inplace=True)
        augmented_df.insert(0, 'id', range(1, len(augmented_df) + 1))
        
        logging.info(f"数据增强完成。原始数据量：{len(df)}，增强后数据量：{len(augmented_df)}")
        
        return {
            'augmented_df': augmented_df,
            'augment_info': augment_info
        }

def augment_dataset(input_file: str, output_file: str, config: AugmentationConfig = None, evaluate: int = 1):
    """
    数据增强主函数
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        config: 配置对象，默认None则使用默认配置
        evaluate: 是否进行评估，1表示进行评估（默认），0表示不进行评估
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置日志
    log_file = os.path.join(output_dir, 'augmentation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 文件处理器
            logging.StreamHandler()  # 控制台处理器
        ]
    )
    logging.info(f"日志将保存到：{log_file}")
    
    # 初始化配置
    config = config or AugmentationConfig()
    if not config.validate():
        raise ValueError("配置验证失败")
    
    # 读取数据
    logging.info(f"读取数据：{input_file}")
    df = pd.read_csv(input_file, sep='|')
    
    # 初始化增强器
    augmenter = TextAugmenter(config)
    
    # 执行增强
    result = augmenter.augment(df)
    augmented_df = result['augmented_df']
    augment_info = result['augment_info']
    
    # 保存总的增强结果
    augmented_df.to_csv(output_file, sep='|', index=False, encoding='utf-8')
    logging.info(f"总的增强结果已保存到：{output_file}")
    
    # 按策略分别保存增强结果
    file_name, file_ext = os.path.splitext(output_file)
    strategy_mapping = {
        'template_replace': '模板替换',
        'word2vec_similar': '词向量相似词替换'
    }
    
    for strategy, strategy_name in strategy_mapping.items():
        # 筛选特定策略的数据
        strategy_df = augmented_df[augmented_df['aug_method'] == strategy]
        if not strategy_df.empty:
            # 构建策略特定的输出文件路径
            strategy_output = f"{file_name}_{strategy}{file_ext}"
            # 保存数据
            strategy_df.to_csv(strategy_output, sep='|', index=False, encoding='utf-8')
            logging.info(f"{strategy_name}结果已保存到：{strategy_output}，数据量：{len(strategy_df)}")
    
    # 如果需要评估
    if evaluate == 1:
        logging.info("开始评估增强效果...")
        evaluator = QualityEvaluator()
        evaluation_file = os.path.join(output_dir, 'evaluation_results.json')
        evaluation_results = {}
        
        for method, info in augment_info.items():
            if info['original']:  # 只评估有数据的方法
                logging.info(f"正在评估{strategy_mapping.get(method, method)}...")
                results = evaluator.evaluate_batch(
                    original_texts=info['original'],
                    augmented_texts=info['augmented'],
                    original_sentiments=info['sentiments'],
                    method=method
                )
                evaluation_results[method] = results
        
        # 保存评估结果
        with open(evaluation_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        logging.info(f"评估结果已保存到：{evaluation_file}")
        
        # 打印评估摘要
        evaluator.print_summary()
    else:
        logging.info("跳过评估步骤")

def main():
    parser = argparse.ArgumentParser(description='汽车评论数据增强工具')
    parser.add_argument('--input_file', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件路径')
    parser.add_argument('--evaluate', type=int, choices=[0, 1], default=1, help='是否进行评估，1表示进行评估（默认），0表示不进行评估')
    
    args = parser.parse_args()
    
    # 确保输出路径使用正确的目录分隔符
    output_file = os.path.normpath(args.output_file)
    
    augment_dataset(args.input_file, output_file, evaluate=args.evaluate)

if __name__ == "__main__":
    main() 