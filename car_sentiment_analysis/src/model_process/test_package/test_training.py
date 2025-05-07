"""
测试模型训练流程
使用少量数据和较少的epoch进行测试训练
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.model_param_config import MODEL_CONFIG
from model_train import CarCommentDataset, CarSentimentModel, train_model
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

def test_training_pipeline():
    print("=== 测试模型训练流程 ===")
    
    # 1. 测试数据加载
    print("\n1. 测试数据加载...")
    try:
        train_data = pd.read_csv('car_sentiment_analysis/data/processed/train.csv', sep='|', encoding='utf-8')
        print(f"数据加载成功! 样本数量: {len(train_data)}")
        print(f"数据列: {train_data.columns.tolist()}")
        print(f"\n标签分布:\n{train_data['sentiment_analysis_results'].value_counts()}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return
    
    # 2. 测试数据集创建
    print("\n2. 测试数据集创建...")
    try:
        # 只使用前100个样本进行测试
        test_size = 100
        tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
        
        train_dataset = CarCommentDataset(
            texts=train_data['content'].values[:test_size],
            likes=train_data['like_count'].values[:test_size],
            replies=train_data['sub_comment_count'].values[:test_size],
            labels=train_data['sentiment_analysis_results'].values[:test_size],
            tokenizer=tokenizer,
            max_length=MODEL_CONFIG['max_length']
        )
        print(f"数据集创建成功! 测试样本数量: {len(train_dataset)}")
        
        # 测试一个batch的数据
        sample = train_dataset[0]
        print(f"\n单个样本数据结构:")
        for k, v in sample.items():
            print(f"{k}: shape {v.shape}")
    except Exception as e:
        print(f"数据集创建失败: {str(e)}")
        return
    
    # 3. 测试模型创建
    print("\n3. 测试模型创建...")
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CarSentimentModel(
            MODEL_CONFIG['model_name'],
            MODEL_CONFIG['num_labels']
        ).to(device)
        print(f"模型创建成功! 使用设备: {device}")
        
        # 测试模型前向传播
        test_batch = next(iter(DataLoader(train_dataset, batch_size=2)))
        with torch.no_grad():
            outputs = model(
                test_batch['input_ids'].to(device),
                test_batch['attention_mask'].to(device),
                test_batch['social_features'].to(device)
            )
        print(f"模型前向传播测试成功! 输出shape: {outputs.shape}")
    except Exception as e:
        print(f"模型创建失败: {str(e)}")
        return
    
    # 4. 测试训练流程
    print("\n4. 测试训练流程...")
    try:
        # 创建小批量数据加载器
        test_dataloader = DataLoader(
            train_dataset,
            batch_size=MODEL_CONFIG['batch_size'],
            shuffle=True
        )
        
        # 修改配置进行快速测试
        test_config = MODEL_CONFIG.copy()
        test_config['epochs'] = 1  # 只训练1个epoch
        
        # 开始测试训练
        history = train_model(test_dataloader, None, model, device, test_config)
        print("训练流程测试完成!")
        print(f"训练历史: {history}")
    except Exception as e:
        print(f"训练流程测试失败: {str(e)}")
        return
    
    print("\n=== 所有测试完成! ===")

if __name__ == "__main__":
    test_training_pipeline() 