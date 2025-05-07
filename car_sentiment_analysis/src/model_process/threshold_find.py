"""
寻找情感分析模型的最优阈值
"""

import os
import torch
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import logging
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from car_sentiment_analysis.config.v4_config import V4_CONFIG
import matplotlib.font_manager as fm

# 设置中文字体
font_path = 'C:/Windows/Fonts/SimHei.ttf'  # 黑体
# 添加字体文件
fm.fontManager.addfont(font_path)
# 设置默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 设置DPI
plt.rcParams['figure.dpi'] = 300
# 设置字体大小
plt.rcParams['font.size'] = 12
# 设置标题字体大小
plt.rcParams['axes.titlesize'] = 14
# 设置坐标轴标签字体大小
plt.rcParams['axes.labelsize'] = 12
# 设置图例字体大小
plt.rcParams['legend.fontsize'] = 10
# 设置刻度标签字体大小
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

# 设置实验相关路径
threshold_range = "0.36-0.46"  # 当前阈值范围
logs_dir = os.path.join(project_root, "src", "saved_models", "logs", "threshold", threshold_range)
os.makedirs(logs_dir, exist_ok=True)  # 确保日志目录存在

# 设置日志文件路径
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(logs_dir, f'threshold_search@{threshold_range}_{timestamp}.log')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info(f"项目根目录: {project_root}")
logging.info(f"日志目录: {logs_dir}")
logging.info(f"日志文件: {log_file}")

# 导入自定义模块
from car_sentiment_analysis.src.model_process.model_train import CarCommentDataset, CarSentimentModel

class ThresholdFinder:
    def __init__(self, model_path, val_dataset):
        """
        初始化阈值查找器
        Args:
            model_path: 模型文件路径
            val_dataset: 验证数据集
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        self.val_loader = DataLoader(val_dataset, batch_size=V4_CONFIG['training']['batch_size'], shuffle=False)
        logging.info(f"使用设备: {self.device}")
        
    def load_model(self, model_path):
        """加载模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model = CarSentimentModel(
                model_name=V4_CONFIG['model_name'],
                num_labels=V4_CONFIG['num_labels']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            logging.info(f"成功加载模型: {model_path}")
            return model
        except Exception as e:
            logging.error(f"模型加载失败: {e}")
            raise
            
    def evaluate_threshold(self, threshold):
        """评估特定阈值的性能"""
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"评估阈值 {threshold:.2f}"):
                # 获取输入数据
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                social_features = batch['social_features'].to(self.device)
                labels = batch['labels'].cpu().numpy()
                
                # 模型预测
                outputs = self.model(input_ids, attention_mask, social_features)
                probs = torch.softmax(outputs, dim=1)
                
                # 应用阈值策略
                positive_probs = probs[:, 1]  # 正向概率
                batch_preds = torch.zeros_like(positive_probs, dtype=torch.long)
                
                # 对整个批次应用阈值
                batch_preds[positive_probs > threshold] = 1  # 正向
                
                # 对非正向的样本，选择其他类别中的最高概率
                non_positive_mask = positive_probs <= threshold
                other_probs = probs.clone()
                other_probs[:, 1] = float('-inf')  # 将正向概率设为负无穷
                other_preds = other_probs.argmax(dim=1)
                batch_preds[non_positive_mask] = other_preds[non_positive_mask]
                
                all_preds.extend(batch_preds.cpu().numpy())
                all_labels.extend(labels)
        
        # 计算整体指标
        metrics = {
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }
        
        # 添加正向类别的单独指标
        positive_metrics = {
            'positive_precision': precision_score(all_labels, all_preds, labels=[1], average='micro'),
            'positive_recall': recall_score(all_labels, all_preds, labels=[1], average='micro'),
            'positive_f1': f1_score(all_labels, all_preds, labels=[1], average='micro')
        }
        
        metrics.update(positive_metrics)
        
        # 记录每个类别的指标
        class_report = classification_report(all_labels, all_preds, output_dict=True)
        logging.info("\n类别详细指标:")
        for i in range(4):
            class_metrics = class_report[str(i)]
            metrics[f'class_{i}_precision'] = class_metrics['precision']
            metrics[f'class_{i}_recall'] = class_metrics['recall']
            metrics[f'class_{i}_f1'] = class_metrics['f1-score']
            logging.info(f"类别 {i}: Precision={class_metrics['precision']:.3f}, "
                        f"Recall={class_metrics['recall']:.3f}, "
                        f"F1={class_metrics['f1-score']:.3f}")
        
        return metrics
        
    def evaluate_baseline(self):
        """评估基线模型（无阈值调整）的性能"""
        logging.info("\n评估基线模型性能（无阈值调整）:")
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="评估基线性能"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                social_features = batch['social_features'].to(self.device)
                labels = batch['labels'].cpu().numpy()
                
                outputs = self.model(input_ids, attention_mask, social_features)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels)
                all_probs.extend(probs.cpu().numpy())
        
        # 计算基线指标
        baseline_metrics = {
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted'),
            'f1': f1_score(all_labels, all_preds, average='weighted')
        }
        
        # 记录基线性能
        logging.info("\n基线模型性能:")
        logging.info(f"整体指标:")
        logging.info(f"  Precision: {baseline_metrics['precision']:.3f}")
        logging.info(f"  Recall: {baseline_metrics['recall']:.3f}")
        logging.info(f"  F1: {baseline_metrics['f1']:.3f}")
        
        # 记录每个类别的详细指标
        class_report = classification_report(all_labels, all_preds, output_dict=True)
        logging.info("\n各类别性能:")
        for i in range(4):
            logging.info(f"类别 {i}: Precision={class_report[str(i)]['precision']:.3f}, "
                        f"Recall={class_report[str(i)]['recall']:.3f}, "
                        f"F1={class_report[str(i)]['f1-score']:.3f}")
        
        return baseline_metrics, all_probs, all_labels

    def plot_threshold_analysis(self, all_results, save_dir):
        """绘制阈值分析图表"""
        # 提取数据
        thresholds = [r['threshold'] for r in all_results]
        
        # 定义所有需要绘制的指标组
        metrics_groups = {
            'precision': {
                '类别0精确率': [r.get('class_0_precision', 0) for r in all_results],
                '类别1精确率': [r.get('class_1_precision', 0) for r in all_results],
                '类别2精确率': [r.get('class_2_precision', 0) for r in all_results],
                '类别3精确率': [r.get('class_3_precision', 0) for r in all_results],
                '整体精确率': [r['overall_precision'] for r in all_results],
            },
            'recall': {
                '类别0召回率': [r.get('class_0_recall', 0) for r in all_results],
                '类别1召回率': [r.get('class_1_recall', 0) for r in all_results],
                '类别2召回率': [r.get('class_2_recall', 0) for r in all_results],
                '类别3召回率': [r.get('class_3_recall', 0) for r in all_results],
                '整体召回率': [r['overall_recall'] for r in all_results],
            },
            'f1': {
                '类别0 F1': [r.get('class_0_f1', 0) for r in all_results],
                '类别1 F1': [r.get('class_1_f1', 0) for r in all_results],
                '类别2 F1': [r.get('class_2_f1', 0) for r in all_results],
                '类别3 F1': [r.get('class_3_f1', 0) for r in all_results],
                '整体F1': [r['overall_f1'] for r in all_results],
            },
            'positive': {
                '正向精确率': [r['positive_precision'] for r in all_results],
                '正向召回率': [r['positive_recall'] for r in all_results],
                '正向F1': [r['positive_f1'] for r in all_results],
            }
        }
        
        # 颜色方案
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC']
        
        # 为每组指标创建单独的图表
        for metric_name, metrics in metrics_groups.items():
            # 创建图表并设置样式
            plt.figure(figsize=(12, 8))
            sns.set_style("whitegrid")
            
            # 绘制每个指标的曲线
            for (name, values), color in zip(metrics.items(), colors):
                plt.plot(thresholds, values, marker='o', label=name, linewidth=2, color=color)
                # 添加数据标签
                for x, y in zip(thresholds, values):
                    plt.annotate(f'{y:.3f}', (x, y), 
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center',
                               fontsize=8,
                               fontproperties='SimHei')
            
            # 添加最优阈值的垂直线
            best_threshold = max(all_results, key=lambda x: x['positive_precision'])['threshold']
            plt.axvline(x=best_threshold, color='gray', linestyle='--', 
                       label=f'最优阈值 ({best_threshold:.2f})')
            
            # 设置图表属性
            title_map = {
                'precision': '精确率对比分析',
                'recall': '召回率对比分析',
                'f1': 'F1分数对比分析',
                'positive': '正向类别指标对比分析'
            }
            
            plt.title(title_map[metric_name], pad=20, fontproperties='SimHei', fontsize=14)
            plt.xlabel('阈值', fontproperties='SimHei', fontsize=12)
            plt.ylabel('指标值', fontproperties='SimHei', fontsize=12)
            
            # 设置图例
            legend = plt.legend(loc='center left', 
                              bbox_to_anchor=(1, 0.5),
                              prop={'family': 'SimHei', 'size': 10})
            
            # 设置网格和范围
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlim(min(thresholds)-0.05, max(thresholds)+0.05)
            plt.ylim(0.5, 1.0)
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图表
            save_path = os.path.join(save_dir, f'threshold_analysis@{threshold_range}_{metric_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"已保存{title_map[metric_name]}图表: {save_path}")

    def find_best_threshold(self):
        """寻找最优阈值"""
        # 首先评估基线性能
        baseline_metrics, baseline_probs, baseline_labels = self.evaluate_baseline()
        
        thresholds = np.arange(0.36, 0.46, 0.02)
        best_metrics = {
            'threshold': 0.5,
            'overall_precision': 0.0,
            'positive_precision': 0.0,
            'overall_f1': 0.0,
            'positive_f1': 0.0,
            'overall_recall': 0.0,
            'positive_recall': 0.0
        }
        
        # 记录所有阈值的结果
        all_results = []
        
        for threshold in thresholds:
            logging.info(f"\n测试阈值: {threshold:.2f}")
            
            metrics = self.evaluate_threshold(threshold)
            
            # 记录结果
            result = {
                'threshold': float(threshold),
                'overall_precision': metrics['precision'],
                'positive_precision': metrics['positive_precision'],
                'overall_f1': metrics['f1'],
                'positive_f1': metrics['positive_f1'],
                'overall_recall': metrics['recall'],
                'positive_recall': metrics['positive_recall'],
                # 添加每个类别的指标
                'class_0_precision': metrics['class_0_precision'],
                'class_1_precision': metrics['class_1_precision'],
                'class_2_precision': metrics['class_2_precision'],
                'class_3_precision': metrics['class_3_precision'],
                'class_0_recall': metrics['class_0_recall'],
                'class_1_recall': metrics['class_1_recall'],
                'class_2_recall': metrics['class_2_recall'],
                'class_3_recall': metrics['class_3_recall'],
                'class_0_f1': metrics['class_0_f1'],
                'class_1_f1': metrics['class_1_f1'],
                'class_2_f1': metrics['class_2_f1'],
                'class_3_f1': metrics['class_3_f1']
            }
            all_results.append(result)
            
            # 更新最佳结果
            current_score = (
                0.6 * metrics['positive_precision'] +
                0.2 * metrics['precision'] +
                0.1 * metrics['positive_f1'] +
                0.1 * metrics['f1']
            )
            
            best_score = (
                0.6 * best_metrics['positive_precision'] +
                0.2 * best_metrics['overall_precision'] +
                0.1 * best_metrics['positive_f1'] +
                0.1 * best_metrics['overall_f1']
            )
            
            if current_score > best_score:
                best_metrics = result
            
            logging.info(f"当前阈值 {threshold:.2f} 的性能:")
            logging.info(f"整体指标:")
            logging.info(f"  Precision: {metrics['precision']:.3f}")
            logging.info(f"  Recall: {metrics['recall']:.3f}")
            logging.info(f"  F1: {metrics['f1']:.3f}")
            logging.info(f"正向类别指标:")
            logging.info(f"  Precision: {metrics['positive_precision']:.3f}")
            logging.info(f"  Recall: {metrics['positive_recall']:.3f}")
            logging.info(f"  F1: {metrics['positive_f1']:.3f}")
            logging.info(f"综合得分: {current_score:.3f}")
        
        # 生成可视化分析
        save_dir = logs_dir
        self.plot_threshold_analysis(all_results, save_dir)
        
        return best_metrics, all_results, baseline_metrics

def main():
    try:
        # 配置路径 - 使用v4模型
        model_path = os.path.join(project_root, 'src/saved_models/checkpoints/v4/best_model.pth')
        
        # 创建tokenizer
        tokenizer = BertTokenizer.from_pretrained(V4_CONFIG['model_name'])
        
        # 加载验证数据
        val_data_path = os.path.join(project_root, 'data/experiments/v4_30000samples/train.csv')
        val_data = pd.read_csv(val_data_path, sep='|')
        
        logging.info(f"加载验证数据: {len(val_data)} 条记录")
        logging.info("\n类别分布:")
        for label, count in val_data['sentiment_analysis_results'].value_counts().items():
            logging.info(f"类别 {label}: {count} 条 ({count/len(val_data)*100:.2f}%)")
        
        # 创建验证数据集
        val_dataset = CarCommentDataset(
            texts=val_data['content'].tolist(),
            labels=val_data['sentiment_analysis_results'].tolist(),
            likes=val_data['like_count'].tolist(),
            replies=val_data['sub_comment_count'].tolist(),
            tokenizer=tokenizer,
            max_length=V4_CONFIG['max_length']
        )
        
        logging.info(f"验证数据集准备完成: {len(val_dataset)} 条记录")
        
        # 创建ThresholdFinder实例
        finder = ThresholdFinder(model_path, val_dataset)
        
        # 寻找最优阈值
        best_metrics, all_results, baseline_metrics = finder.find_best_threshold()
        
        # 保存结果
        config_path = os.path.join(logs_dir, f'threshold_config@{threshold_range}.json')
        config = {
            'baseline_metrics': baseline_metrics,
            'best_threshold': best_metrics['threshold'],
            'best_metrics': {
                'f1': best_metrics['overall_f1'],
                'precision': best_metrics['overall_precision'],
                'recall': best_metrics['overall_recall']
            },
            'all_results': all_results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': 'v4',
            'model_config': {
                'model_name': V4_CONFIG['model_name'],
                'num_labels': V4_CONFIG['num_labels'],
                'max_length': V4_CONFIG['max_length']
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        logging.info(f"\n配置已保存到 {config_path}")
        
    except Exception as e:
        logging.error(f"程序执行错误: {e}")
        raise

if __name__ == "__main__":
    main() 