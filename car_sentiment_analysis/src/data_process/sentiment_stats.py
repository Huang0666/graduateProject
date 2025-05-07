"""
汽车评论情感分析结果统计程序

本程序用于统计汽车评论数据集中不同情感类型的分布情况。
统计内容包括：
1. 四种情感类型（正面、负面、中性、无关）的评论数量
2. 各类型评论占总评论的百分比

数据来源：all_raw_cars_comments_5000.csv
数据格式：CSV文件，使用'|'作为分隔符
情感类型说明：
- 0: 负面评价
- 1: 正面评价
- 2: 中性评价
- 3: 无关评价

输出：
- 在控制台显示统计结果
- 将结果保存到 sentiment_distribution_stats.txt 文件中

作者：[您的名字]
创建日期：[创建日期]
"""

import pandas as pd
import os
from datetime import datetime

def calculate_sentiment_distribution(csv_file):
    """
    计算情感分析结果的分布情况
    
    该函数读取CSV格式的评论数据文件，统计不同情感类型的评论数量和占比。
    
    Args:
        csv_file (str): CSV文件路径，文件应包含'sentiment_analysis_results'列
    
    Returns:
        tuple: 返回两个字典
            - 第一个字典：各情感类型的评论数量
            - 第二个字典：各情感类型占总评论的百分比
    
    示例：
        counts, percentages = calculate_sentiment_distribution("comments.csv")
        # counts 示例: {'正面': 1000, '负面': 500, ...}
        # percentages 示例: {'正面': 50.0, '负面': 25.0, ...}
    """
    # 读取CSV文件，使用'|'作为分隔符
    df = pd.read_csv(csv_file, sep='|')
    
    # 定义情感标签映射：将数字标签映射为中文说明
    sentiment_mapping = {
        0: "负面",
        1: "正面",
        2: "中性",
        3: "无关"
    }
    
    # 计算每种情感的数量
    sentiment_counts = df['sentiment_analysis_results'].value_counts().to_dict()
    
    # 计算评论总数
    total = sum(sentiment_counts.values())
    
    # 计算每种情感类型的百分比
    sentiment_percentages = {k: (v/total * 100) for k, v in sentiment_counts.items()}
    
    # 将数字标签转换为中文标签
    labeled_counts = {sentiment_mapping[k]: v for k, v in sentiment_counts.items()}
    labeled_percentages = {sentiment_mapping[k]: v for k, v in sentiment_percentages.items()}
    
    return labeled_counts, labeled_percentages

def save_results(counts, percentages, output_file):
    """
    将统计结果保存到文件
    
    Args:
        counts (dict): 各情感类型的评论数量
        percentages (dict): 各情感类型的百分比
        output_file (str): 输出文件路径
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("汽车评论情感分析分布统计报告\n")
        f.write(f"生成时间：{current_time}\n")
        f.write("-" * 40 + "\n\n")
        
        # 写入数量统计
        f.write("评论数量统计：\n")
        total = sum(counts.values())
        f.write(f"总评论数：{total}条\n")
        for sentiment, count in counts.items():
            f.write(f"{sentiment}：{count}条\n")
        f.write("\n")
        
        # 写入百分比统计
        f.write("评论类型占比：\n")
        for sentiment, percentage in percentages.items():
            f.write(f"{sentiment}：{percentage:.2f}%\n")

def main():
    """
    主函数：程序入口
    
    功能：
    1. 构建数据文件的路径
    2. 调用统计函数计算分布
    3. 将统计结果保存到文件
    4. 在控制台显示结果摘要
    """
    # 获取数据文件路径：通过相对路径定位到数据文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, '..', '..', 'data', 'experiments', 'v3_6000samples', 'all_data.csv')
    
    # 设置输出文件路径
    output_file = os.path.join(current_dir, 'sentiment_distribution_stats_1.txt')
    
    # 计算情感分布
    counts, percentages = calculate_sentiment_distribution(data_file)
    
    # 保存结果到文件
    save_results(counts, percentages, output_file)
    
    # 在控制台显示结果
    print(f"\n统计结果已保存到：{output_file}")
    print("\n结果摘要:")
    print("-" * 40)
    
    total = sum(counts.values())
    print(f"总评论数：{total}条")
    
    print("\n各类型评论占比:")
    for sentiment, percentage in percentages.items():
        print(f"{sentiment}: {percentage:.2f}%")

if __name__ == "__main__":
    main() 