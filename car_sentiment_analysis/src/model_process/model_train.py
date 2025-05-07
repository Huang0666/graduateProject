"""
统一的模型训练文件，通过配置文件控制不同阶段的训练
"""

import os
import sys

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_recall_curve, auc, roc_curve
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import argparse
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import psutil
import GPUtil
from threading import Thread
import time
from tqdm import tqdm

# 导入配置
from car_sentiment_analysis.config.v1_config import V1_CONFIG
from car_sentiment_analysis.config.v2_config import V2_CONFIG
from car_sentiment_analysis.config.v3_config import V3_CONFIG
from car_sentiment_analysis.config.v4_config import V4_CONFIG

# 添加资源监控类
class ResourceMonitor:
    def __init__(self):
        self.gpu_utils = []
        self.memory_utils = []
        self.timestamps = []
        self.keep_running = True
        
    def start(self):
        self.keep_running = True
        Thread(target=self._monitor_resources).start()
        
    def stop(self):
        self.keep_running = False
        
    def _monitor_resources(self):
        while self.keep_running:
            # GPU使用率
            try:
                gpu = GPUtil.getGPUs()[0]
                self.gpu_utils.append(gpu.load * 100)
            except:
                self.gpu_utils.append(0)
            
            # 内存使用率
            self.memory_utils.append(psutil.virtual_memory().percent)
            self.timestamps.append(time.time())
            time.sleep(1)  # 每秒采样一次

# 添加指标记录类
class MetricsTracker:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.metrics_history = {
            # 性能指标
            'train_accuracy': [],
            'val_accuracy': [],
            'train_loss': [],
            'val_loss': [],
            'f1_scores': [],
            'precision': {i: [] for i in range(self.num_classes)},
            'recall': {i: [] for i in range(self.num_classes)},
            'roc_auc': [],
            
            # 训练过程指标
            'learning_rates': [],
            'gradient_norms': [],
            
            # 资源使用指标
            'gpu_utilization': [],
            'memory_usage': [],
            'timestamps': []
        }
    
    def update_batch_metrics(self, lr, grad_norm):
        self.metrics_history['learning_rates'].append(lr)
        self.metrics_history['gradient_norms'].append(grad_norm)
    
    def update_epoch_metrics(self, train_acc, val_acc, train_loss, val_loss, 
                           f1, precisions, recalls, roc_auc):
        self.metrics_history['train_accuracy'].append(train_acc)
        self.metrics_history['val_accuracy'].append(val_acc)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['f1_scores'].append(f1)
        
        for i in range(self.num_classes):
            self.metrics_history['precision'][i].append(precisions[i])
            self.metrics_history['recall'][i].append(recalls[i])
        
        self.metrics_history['roc_auc'].append(roc_auc)
    
    def update_resource_metrics(self, gpu_util, memory_util, timestamp):
        self.metrics_history['gpu_utilization'].append(gpu_util)
        self.metrics_history['memory_usage'].append(memory_util)
        self.metrics_history['timestamps'].append(timestamp)

def calculate_metrics(outputs, labels, num_classes):
    """计算各项指标"""
    # 转换为numpy数组
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 计算基础指标
    accuracy = accuracy_score(labels, predictions)
    
    # 计算每个类别的precision和recall
    precisions = []
    recalls = []
    for i in range(num_classes):
        binary_labels = (labels == i).astype(int)
        binary_preds = (predictions == i).astype(int)
        
        # 计算precision和recall
        tp = np.sum((binary_labels == 1) & (binary_preds == 1))
        fp = np.sum((binary_labels == 0) & (binary_preds == 1))
        fn = np.sum((binary_labels == 1) & (binary_preds == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # 计算ROC AUC（多类别使用one-vs-rest方式）
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    roc_auc_scores = []
    for i in range(num_classes):
        binary_labels = (labels == i).astype(int)
        try:
            roc_auc = auc(*roc_curve(binary_labels, probs[:, i])[:2])
            roc_auc_scores.append(roc_auc)
        except:
            roc_auc_scores.append(0)
    
    return accuracy, precisions, recalls, np.mean(roc_auc_scores)

def plot_training_metrics(metrics_history, save_dir):
    """绘制训练指标图表，将不同类型的指标分开保存"""
    
    # 1. 损失值和准确率图表
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['train_loss'], label='Train Loss')
    plt.plot(metrics_history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['train_accuracy'], label='Train Accuracy')
    plt.plot(metrics_history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_accuracy.png'))
    plt.close()
    
    # 2. 精确率图表
    plt.figure(figsize=(10, 6))
    for i in range(len(metrics_history['precision'])):
        plt.plot(metrics_history['precision'][i], label=f'Class_{i} Precision')
    plt.title('Precision by Class')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'precision.png'))
    plt.close()
    
    # 3. 召回率图表
    plt.figure(figsize=(10, 6))
    for i in range(len(metrics_history['recall'])):
        plt.plot(metrics_history['recall'][i], label=f'Class_{i} Recall')
    plt.title('Recall by Class')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'recall.png'))
    plt.close()
    
    # 4. F1分数和ROC AUC图表
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['f1_scores'], label='F1 Score')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['roc_auc'], label='ROC AUC')
    plt.title('ROC AUC Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_roc_auc.png'))
    plt.close()
    
    # 5. 学习率和梯度范数图表
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['learning_rates'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['gradient_norms'], label='Gradient Norm')
    plt.title('Gradient Norm')
    plt.xlabel('Step')
    plt.ylabel('L2 Norm')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lr_grad_norm.png'))
    plt.close()
    
    # 6. 资源使用情况图表
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history['gpu_utilization'], label='GPU Usage')
    plt.title('GPU Utilization')
    plt.xlabel('Time')
    plt.ylabel('Utilization %')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history['memory_usage'], label='Memory Usage')
    plt.title('Memory Utilization')
    plt.xlabel('Time')
    plt.ylabel('Utilization %')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'resource_usage.png'))
    plt.close()

class CarCommentDataset(Dataset):
    """汽车评论数据集"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128, likes=None, replies=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.likes = likes if likes is not None else np.zeros(len(texts))
        self.replies = replies if replies is not None else np.zeros(len(texts))
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 添加社交特征
        social_features = torch.tensor(
            [self.likes[idx], self.replies[idx]], 
            dtype=torch.float32
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'social_features': social_features,
            'labels': torch.tensor(label, dtype=torch.long)  # 确保标签是长整型
        }

class CarSentimentModel(nn.Module):
    """汽车评论情感分析模型"""
    
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        # 添加社交特征处理层
        self.social_features_dim = 2  # likes和replies
        self.social_features_layer = nn.Linear(self.social_features_dim, 32)
        
        # 合并BERT输出和社交特征
        self.classifier = nn.Linear(self.bert.config.hidden_size + 32, num_labels)
    
    def forward(self, input_ids, attention_mask, social_features=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 处理社交特征
        if social_features is None:
            social_features = torch.zeros((input_ids.size(0), self.social_features_dim), 
                                       device=input_ids.device)
        social_features = self.social_features_layer(social_features)
        
        # 合并特征
        combined_features = torch.cat([pooled_output, social_features], dim=1)
        logits = self.classifier(combined_features)
        return logits

def load_data(data_path):
    """加载数据
    
    返回:
        texts: 评论内容列表
        labels: 情感标签列表 (0:负面, 1:正面, 2:中性, 3:无关)
        likes: 点赞数列表
        replies: 回复数列表
    """
    df = pd.read_csv(data_path, sep='|', encoding='utf-8')
    
    # 确保sentiment_analysis_results是数值类型
    df['sentiment_analysis_results'] = pd.to_numeric(df['sentiment_analysis_results'])
    
    return (df['content'].values, 
            df['sentiment_analysis_results'].values,
            df['like_count'].values,
            df['sub_comment_count'].values)

def init_new_model(config):
    """初始化新模型"""
    model = CarSentimentModel(
        model_name=config['model_name'],
        num_labels=config['num_labels']
    )
    return model

def load_previous_model(model_path):
    """加载之前训练的模型"""
    checkpoint = torch.load(model_path)
    model = CarSentimentModel(
        model_name=checkpoint['config']['model_name'],
        num_labels=checkpoint['config']['num_labels']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def create_experiment_log(config):
    """创建实验日志"""
    log_file = os.path.join(config['save']['log_dir'], 'experiment_record.md')
    os.makedirs(config['save']['log_dir'], exist_ok=True)
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"# 实验记录：{config['experiment']['description']}\n\n")
        f.write(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 实验配置\n")
        f.write("```python\n")
        f.write(json.dumps(config, indent=2, ensure_ascii=False))
        f.write("\n```\n")
    
    return log_file

def train_model(config):
    """训练模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建实验日志
    log_file = create_experiment_log(config)
    
    # 初始化指标跟踪器和资源监控器
    metrics_tracker = MetricsTracker(config['num_labels'])
    resource_monitor = ResourceMonitor()
    resource_monitor.start()
    
    # 加载数据
    print("正在加载数据...")
    train_texts, train_labels, train_likes, train_replies = load_data(config['data']['train_path'])
    val_texts, val_labels, val_likes, val_replies = load_data(config['data']['val_path'])
    
    # 初始化tokenizer
    print("初始化tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config['model_name'])
    
    # 创建数据集
    train_dataset = CarCommentDataset(train_texts, train_labels, tokenizer, config['max_length'], train_likes, train_replies)
    val_dataset = CarCommentDataset(val_texts, val_labels, tokenizer, config['max_length'], val_likes, val_replies)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # 初始化或加载模型
    if config['data'].get('previous_model_path'):
        print(f"加载已有模型：{config['data']['previous_model_path']}")
        model = load_previous_model(config['data']['previous_model_path'])
    else:
        print("初始化新模型")
        model = init_new_model(config)
    
    model = model.to(device)
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    total_steps = len(train_loader) * config['training']['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config['training']['warmup_ratio']),
        num_training_steps=total_steps
    )
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 混合精度训练
    scaler = GradScaler() if config['fp16'] else None
    
    # 训练循环
    best_f1 = 0
    print(f"\n开始训练，共{config['training']['epochs']}轮...")
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        all_train_outputs = []
        all_train_labels = []
        
        # 添加进度条
        train_pbar = tqdm(train_loader, desc=f'训练 Epoch {epoch+1}', 
                         total=len(train_loader), 
                         ncols=100)
        
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            social_features = batch['social_features'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            if config['fp16']:
                with autocast():
                    outputs = model(input_ids, attention_mask, social_features)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask, social_features)
                loss = criterion(outputs, labels)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            train_loss += loss.item()
            
            # 更新进度条
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 记录batch级别的指标
            metrics_tracker.update_batch_metrics(
                scheduler.get_last_lr()[0],
                grad_norm.item()
            )
            
            all_train_outputs.append(outputs.detach())
            all_train_labels.append(labels)
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 计算训练集指标
        train_outputs = torch.cat(all_train_outputs)
        train_labels = torch.cat(all_train_labels)
        train_accuracy, train_precisions, train_recalls, train_roc_auc = calculate_metrics(
            train_outputs, train_labels, config['num_labels']
        )
        
        # 验证阶段
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        all_val_outputs = []
        all_val_labels = []
        
        # 添加验证进度条
        val_pbar = tqdm(val_loader, desc=f'验证 Epoch {epoch+1}', 
                       total=len(val_loader), 
                       ncols=100)
        
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                social_features = batch['social_features'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, social_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # 更新验证进度条
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # 保存用于计算F1分数
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
                # 保存用于计算其他指标
                all_val_outputs.append(outputs)
                all_val_labels.append(labels)
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 计算F1分数
        val_f1 = f1_score(true_labels, predictions, average='weighted')
        
        # 计算验证集其他指标
        val_outputs = torch.cat(all_val_outputs)
        val_labels = torch.cat(all_val_labels)
        val_accuracy, val_precisions, val_recalls, val_roc_auc = calculate_metrics(
            val_outputs, val_labels, config['num_labels']
        )
        
        # 更新epoch级别的指标
        metrics_tracker.update_epoch_metrics(
            train_accuracy, val_accuracy,
            avg_train_loss, avg_val_loss,
            val_f1,  # 使用原有的F1分数
            val_precisions, val_recalls,
            val_roc_auc
        )
        
        # 更新资源使用指标
        if resource_monitor.timestamps:
            metrics_tracker.update_resource_metrics(
                resource_monitor.gpu_utils[-1],
                resource_monitor.memory_utils[-1],
                resource_monitor.timestamps[-1]
            )
        
        # 保存最佳模型（仍然使用F1分数作为标准）
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(config['save']['checkpoint_dir'], exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'config': config,
                'metrics_history': metrics_tracker.metrics_history
            }, os.path.join(config['save']['checkpoint_dir'], 'best_model.pth'))
        
        # 记录训练信息（同时包含原有指标和新增指标）
        print(f"训练损失：{avg_train_loss:.4f}")
        print(f"验证损失：{avg_val_loss:.4f}")
        print(f"验证F1：{val_f1:.4f}")  # 保持原有的F1分数输出
        print(f"训练准确率：{train_accuracy:.4f}")
        print(f"验证准确率：{val_accuracy:.4f}")
        print(f"ROC AUC：{val_roc_auc:.4f}")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n### Epoch {epoch+1}\n")
            f.write(f"- 训练损失：{avg_train_loss:.4f}\n")
            f.write(f"- 验证损失：{avg_val_loss:.4f}\n")
            f.write(f"- 验证F1：{val_f1:.4f}\n")  # 保持原有的F1分数记录
            f.write(f"- 训练准确率：{train_accuracy:.4f}\n")
            f.write(f"- 验证准确率：{val_accuracy:.4f}\n")
            f.write(f"- ROC AUC：{val_roc_auc:.4f}\n")
    
    # 停止资源监控
    resource_monitor.stop()
    
    # 绘制训练指标图表
    plot_training_metrics(metrics_tracker.metrics_history, config['save']['log_dir'])
    
    return {
        'best_f1': best_f1,  # 保持原有的返回值
        'metrics_history': metrics_tracker.metrics_history  # 添加新的指标历史
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, required=True, 
                      choices=['v1', 'v2', 'v3', 'v4'],
                      help='选择实验版本：v1(1000条), v2(3000条), v3(6000条), v4(30000条)')
    
    args = parser.parse_args()
    
    # 根据版本选择配置
    config_map = {
        'v1': V1_CONFIG,
        'v2': V2_CONFIG,
        'v3': V3_CONFIG,
        'v4': V4_CONFIG
    }
    
    config = config_map[args.version]
    print(f"开始{config['experiment']['description']}")
    print(f"实验重点：{config['experiment']['focus']}")
    
    # 开始训练
    results = train_model(config)
    
    # 打印结果
    print("\n训练完成！")
    print(f"最佳验证集F1分数：{results['best_f1']:.4f}")
    print(f"模型保存在：{config['save']['checkpoint_dir']}")

if __name__ == '__main__':
    main() 