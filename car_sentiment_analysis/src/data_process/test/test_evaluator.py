"""
评估器测试模块
"""

import unittest
import pandas as pd
from pathlib import Path
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_process.quality_evaluator import QualityEvaluator
from config.evaluation_config import EvaluationConfig

class TestQualityEvaluator(unittest.TestCase):
    """评估器测试类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.config = EvaluationConfig()
        cls.evaluator = QualityEvaluator()
        
        # 测试数据
        cls.test_data = {
            'original_texts': [
                "丰田卡罗拉：发动机动力很充沛，起步快，油耗也不错。",
                "本田思域：外观设计很运动，内饰做工精致，就是价格贵了点。",
                "大众速腾：底盘调教不错，操控性好，但是发动机噪音有点大。"
            ],
            'augmented_texts': [
                "丰田卡罗拉：引擎动力十足，加速快，油耗很省。",
                "本田思域：造型很时尚，内部用料讲究，就是售价偏高。",
                "大众速腾：悬挂调校优秀，驾驶感很好，但是引擎声音略大。"
            ],
            'original_sentiments': [1, 2, 0]  # 1:正面, 2:中性, 0:负面
        }
        
    def test_semantic_preservation(self):
        """测试语义保持率评估"""
        rate, results = self.evaluator.evaluate_semantic_preservation(
            self.test_data['original_texts'],
            self.test_data['augmented_texts']
        )
        self.assertIsInstance(rate, float)
        self.assertTrue(0 <= rate <= 1)
        self.assertEqual(len(results), len(self.test_data['original_texts']))
        
    def test_sentiment_consistency(self):
        """测试情感一致性评估"""
        rate, results = self.evaluator.evaluate_sentiment_consistency(
            self.test_data['original_sentiments'],
            self.test_data['augmented_texts']
        )
        self.assertIsInstance(rate, float)
        self.assertTrue(0 <= rate <= 1)
        self.assertEqual(len(results), len(self.test_data['original_texts']))
        
    def test_domain_relevance(self):
        """测试领域相关性评估"""
        rate, scores = self.evaluator.evaluate_domain_relevance(
            self.test_data['augmented_texts']
        )
        self.assertIsInstance(rate, float)
        self.assertTrue(0 <= rate <= 1)
        self.assertEqual(len(scores), len(self.test_data['augmented_texts']))
        
    def test_batch_evaluation(self):
        """测试批量评估"""
        results = self.evaluator.evaluate_batch(
            original_texts=self.test_data['original_texts'],
            augmented_texts=self.test_data['augmented_texts'],
            original_sentiments=self.test_data['original_sentiments'],
            method='test_method'
        )
        
        # 检查结果格式
        self.assertIn('method', results)
        self.assertIn('sample_count', results)
        self.assertIn('semantic_preservation', results)
        self.assertIn('sentiment_consistency', results)
        self.assertIn('domain_relevance', results)
        
        # 检查数值
        self.assertEqual(results['sample_count'], len(self.test_data['original_texts']))
        self.assertTrue(0 <= results['semantic_preservation']['rate'] <= 1)
        self.assertTrue(0 <= results['sentiment_consistency']['rate'] <= 1)
        self.assertTrue(0 <= results['domain_relevance']['rate'] <= 1)
        
    def test_save_and_print(self):
        """测试结果保存和打印"""
        # 执行一次评估
        self.evaluator.evaluate_batch(
            original_texts=self.test_data['original_texts'],
            augmented_texts=self.test_data['augmented_texts'],
            original_sentiments=self.test_data['original_sentiments'],
            method='test_method'
        )
        
        # 测试保存结果
        test_output = 'test_evaluation_results.json'
        self.evaluator.save_results(test_output)
        self.assertTrue(os.path.exists(test_output))
        
        # 清理测试文件
        os.remove(test_output)
        
        # 测试打印摘要（不会实际检查输出，但确保方法可以执行）
        try:
            self.evaluator.print_summary()
        except Exception as e:
            self.fail(f"打印摘要失败：{e}")

if __name__ == '__main__':
    unittest.main() 