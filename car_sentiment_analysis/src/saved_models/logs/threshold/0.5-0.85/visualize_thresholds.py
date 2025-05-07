import json
import matplotlib.pyplot as plt
import os

# 读取JSON文件
def load_threshold_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_visualization(data):
    # 提取数据
    thresholds = [result['threshold'] for result in data['all_results']]
    metrics = {
        'overall_precision': [result['overall_precision'] for result in data['all_results']],
        'positive_precision': [result['positive_precision'] for result in data['all_results']],
        'overall_f1': [result['overall_f1'] for result in data['all_results']],
        'positive_f1': [result['positive_f1'] for result in data['all_results']],
        'overall_recall': [result['overall_recall'] for result in data['all_results']],
        'positive_recall': [result['positive_recall'] for result in data['all_results']]
    }

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图1：Precision对比
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, metrics['overall_precision'], marker='o', label='整体精确率')
    plt.plot(thresholds, metrics['positive_precision'], marker='s', label='正例精确率')
    plt.xlabel('阈值')
    plt.ylabel('精确率')
    plt.title('精确率随阈值变化')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_precision.png')
    plt.close()

    # 创建图2：F1-score对比
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, metrics['overall_f1'], marker='o', label='整体F1')
    plt.plot(thresholds, metrics['positive_f1'], marker='s', label='正例F1')
    plt.xlabel('阈值')
    plt.ylabel('F1分数')
    plt.title('F1分数随阈值变化')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_f1.png')
    plt.close()

    # 创建图3：Recall对比
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, metrics['overall_recall'], marker='o', label='整体召回率')
    plt.plot(thresholds, metrics['positive_recall'], marker='s', label='正例召回率')
    plt.xlabel('阈值')
    plt.ylabel('召回率')
    plt.title('召回率随阈值变化')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_recall.png')
    plt.close()

    # 创建图4：所有整体指标对比
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, metrics['overall_precision'], marker='o', label='整体精确率')
    plt.plot(thresholds, metrics['overall_f1'], marker='s', label='整体F1')
    plt.plot(thresholds, metrics['overall_recall'], marker='^', label='整体召回率')
    plt.xlabel('阈值')
    plt.ylabel('指标值')
    plt.title('整体指标随阈值变化')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_overall_metrics.png')
    plt.close()

    # 创建图5：所有正例指标对比
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, metrics['positive_precision'], marker='o', label='正例精确率')
    plt.plot(thresholds, metrics['positive_f1'], marker='s', label='正例F1')
    plt.plot(thresholds, metrics['positive_recall'], marker='^', label='正例召回率')
    plt.xlabel('阈值')
    plt.ylabel('指标值')
    plt.title('正例指标随阈值变化')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_positive_metrics.png')
    plt.close()

if __name__ == "__main__":
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, 'threshold_config.json')
    
    # 加载数据并创建可视化
    data = load_threshold_data(json_path)
    create_visualization(data)
    print("可视化图表已生成完成！") 