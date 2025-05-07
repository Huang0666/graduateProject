"""
准备第三版训练数据的预处理脚本（使用全量数据）

处理步骤：
1. 读取原始数据（全量6015条）
2. 保持原始类别分布
3. 划分训练集和验证集（8:2）
4. 保存数据统计信息
"""

import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split

def create_data_stats(df, train_df, val_df):
    """创建数据统计信息"""
    stats = {
        'total': {
            'total_samples': len(df),
            'class_distribution': df['sentiment_analysis_results'].value_counts().to_dict(),
            'class_percentages': (df['sentiment_analysis_results'].value_counts(normalize=True) * 100).to_dict()
        },
        'train': {
            'total_samples': len(train_df),
            'class_distribution': train_df['sentiment_analysis_results'].value_counts().to_dict(),
            'class_percentages': (train_df['sentiment_analysis_results'].value_counts(normalize=True) * 100).to_dict()
        },
        'val': {
            'total_samples': len(val_df),
            'class_distribution': val_df['sentiment_analysis_results'].value_counts().to_dict(),
            'class_percentages': (val_df['sentiment_analysis_results'].value_counts(normalize=True) * 100).to_dict()
        }
    }
    return stats

def prepare_v3_data(input_file, output_dir):
    """准备第三版训练数据（使用全量数据）"""
    print("\n开始数据预处理...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始数据
    print(f"\n1. 读取原始数据: {input_file}")
    df = pd.read_csv(input_file, sep='|')
    print(f"总数据量: {len(df)}条")
    
    # 情感标签映射（仅用于显示）
    sentiment_map = {
        0: '负面',
        1: '正面',
        2: '中性',
        3: '无关'
    }
    
    # 显示原始数据的类别分布
    print("\n原始数据情感分布:")
    original_dist = df['sentiment_analysis_results'].value_counts()
    for label, count in original_dist.items():
        print(f"{sentiment_map[label]}: {count}条 ({count/len(df)*100:.2f}%)")
    
    # 划分训练集和验证集
    print("\n2. 划分训练集和验证集 (80:20)")
    train_data, val_data = train_test_split(
        df, 
        test_size=0.2,
        random_state=42,
        stratify=df['sentiment_analysis_results']  # 确保划分后保持原有的类别分布
    )
    
    print(f"\n训练集数量: {len(train_data)}条")
    print(f"验证集数量: {len(val_data)}条")
    
    # 保存数据
    print(f"\n3. 保存处理后的数据到: {output_dir}")
    train_file = os.path.join(output_dir, 'train.csv')
    val_file = os.path.join(output_dir, 'val.csv')
    all_file = os.path.join(output_dir, 'all_data.csv')
    stats_file = os.path.join(output_dir, 'data_stats.json')
    
    # 保存训练集和验证集
    train_data.to_csv(train_file, sep='|', index=False)
    val_data.to_csv(val_file, sep='|', index=False)
    
    # 合并并保存全部数据
    all_data = pd.concat([train_data, val_data], axis=0)
    all_data.to_csv(all_file, sep='|', index=False)
    
    # 创建并保存数据统计信息
    stats = create_data_stats(df, train_data, val_data)
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n文件已保存:")
    print(f"- 训练集: {train_file}")
    print(f"- 验证集: {val_file}")
    print(f"- 全部数据: {all_file}")
    print(f"- 统计信息: {stats_file}")
    
    return stats

if __name__ == '__main__':
    input_file = '../../data/raw/all_raw_cars_comments_5000.csv'
    output_dir = '../../data/experiments/v3_6000samples'
    stats = prepare_v3_data(input_file, output_dir)
    
    # 打印最终数据分布
    print("\n最终数据集分布情况:")
    print("\n训练集:")
    print(f"总样本数：{stats['train']['total_samples']}条")
    print("类别分布：")
    for label, count in stats['train']['class_distribution'].items():
        percentage = stats['train']['class_percentages'][label]
        print(f"{label}: {count}条 ({percentage:.2f}%)")
    
    print("\n验证集:")
    print(f"总样本数：{stats['val']['total_samples']}条")
    print("类别分布：")
    for label, count in stats['val']['class_distribution'].items():
        percentage = stats['val']['class_percentages'][label]
        print(f"{label}: {count}条 ({percentage:.2f}%)")
