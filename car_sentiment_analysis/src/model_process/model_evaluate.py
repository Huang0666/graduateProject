"""
基于BERT的汽车评论多维度情感分析模型评估

用于在测试集上评估模型性能,包括:
1. 准确率、精确率、召回率、F1分数
2. 混淆矩阵
3. 各类别的详细评估指标
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from config.model_param_config import MODEL_CONFIG
from model_train import CarCommentDataset, CarSentimentModel

def load_model(model_path):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path)
    
    model = CarSentimentModel(
        MODEL_CONFIG['model_name'],
        MODEL_CONFIG['num_labels']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['config']

def evaluate_model(model, test_dataloader, device):
    """评估模型性能"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            social_features = batch['social_features'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask, social_features)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.numpy())
    
    return np.array(predictions), np.array(true_labels)

def plot_confusion_matrix(true_labels, predictions, class_names, output_dir):
    """绘制混淆矩阵"""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    # 创建预测结果目录
    os.makedirs(MODEL_CONFIG['prediction_dir'], exist_ok=True)
    
    # 加载测试数据
    test_data = pd.read_csv('../data/processed/test.csv', sep='|', encoding='utf-8')
    
    # 加载模型
    model_path = os.path.join(MODEL_CONFIG['checkpoint_dir'], 'best_model.pth')
    if not os.path.exists(model_path):
        # 如果找不到best_model.pth，查找F1分数最高的检查点
        checkpoints = [f for f in os.listdir(MODEL_CONFIG['checkpoint_dir']) 
                      if f.startswith('best_model_f1_')]
        if checkpoints:
            # 按F1分数排序
            best_checkpoint = sorted(checkpoints, 
                                  key=lambda x: float(x.split('_f1_')[1].replace('.pth', '')),
                                  reverse=True)[0]
            model_path = os.path.join(MODEL_CONFIG['checkpoint_dir'], best_checkpoint)
        else:
            raise FileNotFoundError("No model checkpoint found!")
    
    print(f"Loading model from: {model_path}")
    model, trained_config = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    
    # 创建测试数据集
    test_dataset = CarCommentDataset(
        texts=test_data['content'].values,
        likes=test_data['like_count'].values,
        replies=test_data['sub_comment_count'].values,
        labels=test_data['sentiment_analysis_results'].values,
        tokenizer=tokenizer,
        max_length=MODEL_CONFIG['max_length']
    )
    
    # 创建测试数据加载器
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=False
    )
    
    # 评估模型
    predictions, true_labels = evaluate_model(model, test_dataloader, device)
    
    # 计算评估指标
    class_names = ['负面', '正面', '中性', '无关']
    
    # 生成评估报告
    report = classification_report(true_labels, predictions, 
                                target_names=class_names)
    
    # 计算F1分数
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    # 保存评估结果
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(MODEL_CONFIG['prediction_dir'], f'evaluation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存评估报告
    with open(os.path.join(results_dir, 'evaluation_report.txt'), 'w', 
              encoding='utf-8') as f:
        f.write(f"Model path: {model_path}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nWeighted F1 Score: {f1:.4f}\n")
    
    # 绘制混淆矩阵
    plot_confusion_matrix(true_labels, predictions, class_names, results_dir)
    
    # 保存详细结果
    results = {
        'model_path': model_path,
        'predictions': predictions.tolist(),
        'true_labels': true_labels.tolist(),
        'f1_score': f1,
        'classification_report': report,
        'test_samples': len(test_dataset),
        'timestamp': timestamp
    }
    
    # 将结果保存为JSON文件
    # 更改evaluation_results.json文件名 修改保存路径
    import json
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w', 
              encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nEvaluation results saved to: {results_dir}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)

if __name__ == '__main__':
    main()
