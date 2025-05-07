"""
准备第二版训练数据（3000条）的预处理脚本

处理步骤：
1. 读取原始数据
2. 计算评论质量权重（基于点赞数和回复数）
3. 按类别加权采样（每类750条）
4. 划分训练集和验证集（8:2）
5. 保存数据统计信息
"""

import os
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split

# 情感标签映射（仅用于显示）
sentiment_map = {
    0: '负面',
    1: '正面',
    2: '中性',
    3: '无关'
}

def calculate_quality_weight(row):
    """
    计算评论质量权重
    权重 = 1 + log(1 + 点赞数) + log(1 + 回复数)
    使用log是为了平滑极端值的影响
    """
    return 1 + np.log1p(row['like_count']) + np.log1p(row['sub_comment_count'])

def create_data_stats(df, train_df, val_df):
    """创建数据统计信息"""
    stats = {
        'total': {
            'total_samples': len(df),
            'class_distribution': df['sentiment_analysis_results'].value_counts().to_dict()
        },
        'train': {
            'total_samples': len(train_df),
            'class_distribution': train_df['sentiment_analysis_results'].value_counts().to_dict(),
            'avg_likes': train_df['like_count'].mean(),
            'avg_replies': train_df['sub_comment_count'].mean()
        },
        'val': {
            'total_samples': len(val_df),
            'class_distribution': val_df['sentiment_analysis_results'].value_counts().to_dict(),
            'avg_likes': val_df['like_count'].mean(),
            'avg_replies': val_df['sub_comment_count'].mean()
        }
    }
    return stats

def prepare_v2_data(input_file, output_dir, sample_size=3000):
    """准备第二版训练数据（考虑评论质量的加权采样）"""
    print("\n开始数据预处理...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始数据
    print(f"\n1. 读取原始数据: {input_file}")
    df = pd.read_csv(input_file, sep='|')
    print(f"原始数据总量: {len(df)}条")
    
    # 显示原始数据的类别分布
    print("\n原始数据情感分布:")
    original_dist = df['sentiment_analysis_results'].value_counts()
    for label, count in original_dist.items():
        print(f"{sentiment_map[label]}: {count}条 ({count/len(df)*100:.2f}%)")
    
    # 计算评论质量权重
    print("\n2. 计算评论质量权重")
    df['quality_weight'] = df.apply(calculate_quality_weight, axis=1)
    
    # 按类别加权采样
    print(f"\n3. 开始加权采样 (目标每类{sample_size//4}条)")
    samples_per_class = sample_size // 4
    balanced_data = pd.DataFrame()
    
    for label, label_name in sentiment_map.items():
        class_data = df[df['sentiment_analysis_results'] == label]
        if len(class_data) >= samples_per_class:
            # 使用quality_weight作为采样权重
            sampled = class_data.sample(
                n=samples_per_class,
                weights='quality_weight',
                random_state=42
            )
            print(f"{label_name}: 成功采样{samples_per_class}条")
            print(f"平均点赞数: {sampled['like_count'].mean():.2f}")
            print(f"平均回复数: {sampled['sub_comment_count'].mean():.2f}")
        else:
            print(f"警告：{label_name}类别数据不足{samples_per_class}条，实际只有{len(class_data)}条")
            sampled = class_data
        balanced_data = pd.concat([balanced_data, sampled])
    
    print(f"\n采样后总数据量: {len(balanced_data)}条")
    
    # 划分训练集和验证集
    print("\n4. 划分训练集和验证集 (80:20)")
    train_data, val_data = train_test_split(
        balanced_data, 
        test_size=0.2,
        random_state=42,
        stratify=balanced_data['sentiment_analysis_results']
    )
    
    # 保存数据
    print(f"\n5. 保存处理后的数据到: {output_dir}")
    train_file = os.path.join(output_dir, 'train.csv')
    val_file = os.path.join(output_dir, 'val.csv')
    all_file = os.path.join(output_dir, 'all_data.csv')
    stats_file = os.path.join(output_dir, 'data_stats.json')
    
    # 删除quality_weight列后保存
    train_data = train_data.drop('quality_weight', axis=1)
    val_data = val_data.drop('quality_weight', axis=1)
    
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
    output_dir = '../../data/experiments/v2_3000samples'
    stats = prepare_v2_data(input_file, output_dir)
    
    # 打印最终数据分布
    print("\n最终数据集分布情况:")
    print("\n训练集:")
    print(f"总样本数：{stats['train']['total_samples']}条")
    print("类别分布：")
    for label, count in stats['train']['class_distribution'].items():
        print(f"{sentiment_map[label]}: {count}条 ({count/stats['train']['total_samples']*100:.2f}%)")
    print(f"平均点赞数：{stats['train']['avg_likes']:.2f}")
    print(f"平均回复数：{stats['train']['avg_replies']:.2f}")
    
    print("\n验证集:")
    print(f"总样本数：{stats['val']['total_samples']}条")
    print("类别分布：")
    for label, count in stats['val']['class_distribution'].items():
        print(f"{sentiment_map[label]}: {count}条 ({count/stats['val']['total_samples']*100:.2f}%)")
    print(f"平均点赞数：{stats['val']['avg_likes']:.2f}")
    print(f"平均回复数：{stats['val']['avg_replies']:.2f}")
