"""
准备第四版训练数据（30000条）的预处理脚本

处理步骤：
1. 读取增强后的数据
2. 按情感类别分层抽样（保持原始分布）
3. 每个类别按7:2:1划分
4. 合并数据集并保存统计信息
"""

import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split

def print_column_names(df):
    """打印数据框的列名"""
    print("\n数据列名:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    print()

def quality_check(df):
    """数据质量检查"""
    # 首先打印列名，以便调试
    print_column_names(df)
    
    quality_stats = {
        'missing_values': df.isnull().sum().to_dict(),
        # 暂时注释掉可能导致错误的检查，等确认列名后再恢复
        # 'duplicate_comments': df['comment_content'].duplicated().sum(),
        # 'avg_comment_length': df['comment_content'].str.len().mean(),
        # 'short_comments': df[df['comment_content'].str.len() < 10].shape[0],
        # 'long_comments': df[df['comment_content'].str.len() > 500].shape[0]
    }
    return quality_stats

def create_detailed_stats(train_df, val_df, test_df):
    """创建详细的数据统计信息"""
    sentiment_map = {
        0: '负面',
        1: '正面',
        2: '中性',
        3: '无关'
    }
    
    stats = {
        'train': {
            'total_samples': len(train_df),
            'class_distribution': train_df['sentiment_analysis_results'].value_counts().to_dict(),
            'class_percentages': (train_df['sentiment_analysis_results'].value_counts(normalize=True) * 100).to_dict(),
            'quality_metrics': quality_check(train_df)
        },
        'val': {
            'total_samples': len(val_df),
            'class_distribution': val_df['sentiment_analysis_results'].value_counts().to_dict(),
            'class_percentages': (val_df['sentiment_analysis_results'].value_counts(normalize=True) * 100).to_dict(),
            'quality_metrics': quality_check(val_df)
        },
        'test': {
            'total_samples': len(test_df),
            'class_distribution': test_df['sentiment_analysis_results'].value_counts().to_dict(),
            'class_percentages': (test_df['sentiment_analysis_results'].value_counts(normalize=True) * 100).to_dict(),
            'quality_metrics': quality_check(test_df)
        }
    }
    
    # 添加可读的类别名称
    for split in ['train', 'val', 'test']:
        readable_dist = {}
        readable_percent = {}
        for label, count in stats[split]['class_distribution'].items():
            readable_dist[sentiment_map[label]] = count
            readable_percent[sentiment_map[label]] = stats[split]['class_percentages'][label]
        stats[split]['readable_distribution'] = readable_dist
        stats[split]['readable_percentages'] = readable_percent
    
    return stats

def save_and_verify_data(train_data, val_data, test_data, output_dir):
    """保存数据并验证完整性"""
    # 保存数据
    train_file = os.path.join(output_dir, 'train.csv')
    val_file = os.path.join(output_dir, 'val.csv')
    test_file = os.path.join(output_dir, 'test.csv')
    
    train_data.to_csv(train_file, sep='|', index=False)
    val_data.to_csv(val_file, sep='|', index=False)
    test_data.to_csv(test_file, sep='|', index=False)
    
    # 验证数据完整性
    train_verify = pd.read_csv(train_file, sep='|')
    val_verify = pd.read_csv(val_file, sep='|')
    test_verify = pd.read_csv(test_file, sep='|')
    
    assert len(train_verify) == len(train_data), "训练集保存出现问题"
    assert len(val_verify) == len(val_data), "验证集保存出现问题"
    assert len(test_verify) == len(test_data), "测试集保存出现问题"
    
    print(f"\n数据已保存并验证:")
    print(f"- 训练集: {train_file} ({len(train_data)}条)")
    print(f"- 验证集: {val_file} ({len(val_data)}条)")
    print(f"- 测试集: {test_file} ({len(test_data)}条)")

def prepare_v4_data(input_file, output_dir):
    """准备第四版训练数据（大规模数据）"""
    print("\n开始数据预处理...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取数据
    print(f"\n1. 读取数据: {input_file}")
    df = pd.read_csv(input_file, sep='|')
    print(f"总数据量: {len(df)}条")
    
    # 打印列名
    print_column_names(df)
    
    # 按情感类别分层处理
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    sentiment_map = {0: '负面', 1: '正面', 2: '中性', 3: '无关'}
    print("\n2. 按情感类别分层划分数据")
    
    for sentiment in df['sentiment_analysis_results'].unique():
        # 获取当前情感类别的数据
        sentiment_data = df[df['sentiment_analysis_results'] == sentiment]
        print(f"\n处理{sentiment_map[sentiment]}类别 (共{len(sentiment_data)}条):")
        
        # 第一次划分：分出测试集（10%）
        temp_data, test_split = train_test_split(
            sentiment_data,
            test_size=0.1,
            random_state=42
        )
        
        # 第二次划分：剩余数据分为训练集（70%）和验证集（20%）
        train_split, val_split = train_test_split(
            temp_data,
            test_size=0.22,  # 0.22 约等于 20/(70+20)
            random_state=42
        )
        
        print(f"- 训练集: {len(train_split)}条")
        print(f"- 验证集: {len(val_split)}条")
        print(f"- 测试集: {len(test_split)}条")
        
        # 合并数据
        train_data = pd.concat([train_data, train_split])
        val_data = pd.concat([val_data, val_split])
        test_data = pd.concat([test_data, test_split])
    
    # 打乱数据顺序
    print("\n3. 打乱数据顺序")
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 创建统计信息
    print("\n4. 创建数据统计信息")
    stats = create_detailed_stats(train_data, val_data, test_data)
    
    # 保存数据和统计信息
    print("\n5. 保存数据和统计信息")
    save_and_verify_data(train_data, val_data, test_data, output_dir)
    
    # 保存统计信息
    stats_file = os.path.join(output_dir, 'data_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"- 统计信息: {stats_file}")
    
    # 打印数据分布情况
    print("\n最终数据集分布情况:")
    for split in ['train', 'val', 'test']:
        print(f"\n{split.capitalize()}集:")
        print(f"总样本数：{stats[split]['total_samples']}条")
        print("类别分布：")
        for label, percentage in stats[split]['readable_percentages'].items():
            count = stats[split]['readable_distribution'][label]
            print(f"{label}: {count}条 ({percentage:.2f}%)")
    
    return train_data, val_data, test_data, stats

if __name__ == '__main__':
    input_file = '../../data/augmented/augmented_comments_30000_new.csv'
    output_dir = '../../data/experiments/v4_30000samples'
    
    # 执行数据处理
    train_data, val_data, test_data, stats = prepare_v4_data(input_file, output_dir) 