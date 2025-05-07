"""
使用示例数据测试评估功能
"""

import pandas as pd
from pathlib import Path
import sys
import logging
import os

# 添加项目根目录到系统路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_process.quality_evaluator import QualityEvaluator
from config.evaluation_config import EvaluationConfig

def create_sample_data():
    """创建示例数据"""
    # 原始数据
    original_data = pd.DataFrame({
        'id': range(1, 11),
        'car_name': ['丰田卡罗拉', '本田思域', '大众速腾'] * 3 + ['奥迪A4L'],
        'content': [
            '丰田卡罗拉：发动机动力很充沛，起步快，油耗也不错。',
            '本田思域：外观设计很运动，内饰做工精致，就是价格贵了点。',
            '大众速腾：底盘调教不错，操控性好，但是发动机噪音有点大。',
            '丰田卡罗拉：变速箱换挡顺畅，动力输出线性，很省油。',
            '本田思域：车身设计时尚，空间利用率高，就是价位偏高。',
            '大众速腾：悬挂偏硬，方向盘反馈清晰，发动机声音有点吵。',
            '丰田卡罗拉：双擎系统节能环保，起步平顺，就是价格贵。',
            '本田思域：中控台布局合理，做工细腻，性价比不错。',
            '大众速腾：转向精准，制动有力，动力表现一般。',
            '奥迪A4L：科技配置丰富，内饰精致，就是保养贵。'
        ],
        'sentiment_analysis_results': [1, 2, 0, 1, 2, 0, 2, 1, 2, 2]  # 1:正面, 2:中性, 0:负面
    })
    
    # 增强数据
    augmented_data = pd.DataFrame({
        'id': range(1, 31),
        'car_name': ['丰田卡罗拉', '本田思域', '大众速腾'] * 9 + ['奥迪A4L'] * 3,
        'content': [
            # 模板替换
            '丰田卡罗拉：引擎动力十足，提速快，油耗很省。',
            '本田思域：造型很时尚，内部用料讲究，就是售价偏高。',
            '大众速腾：底盘调校优秀，驾驶感很好，但是引擎声音略大。',
            '丰田卡罗拉：变速器平顺，动力输出连贯，节能出色。',
            '本田思域：外形设计前卫，空间宽敞，性价比一般。',
            '大众速腾：减震偏硬，转向反馈清楚，动力噪音明显。',
            '丰田卡罗拉：混动系统高效，起步安静，售价不低。',
            '本田思域：内饰布局人性化，工艺精良，总体不错。',
            '大众速腾：操控精确，刹车给力，动力一般般。',
            '奥迪A4L：配置先进，用料考究，维修费用高。',
            
            # 同义词替换
            '丰田卡罗拉：引擎性能充沛，提速迅速，能耗优异。',
            '本田思域：外形运动，装潢精美，价格偏贵。',
            '大众速腾：底盘优秀，驾控出色，但是发动机声浪大。',
            '丰田卡罗拉：变速箱顺滑，动力输出均匀，省油出色。',
            '本田思域：外观动感，空间实用，定价偏高。',
            '大众速腾：悬架硬朗，方向精准，机械噪音明显。',
            '丰田卡罗拉：混合动力节能，起步轻柔，价格昂贵。',
            '本田思域：中控实用，品质出众，性价比可以。',
            '大众速腾：转向灵敏，制动可靠，动力一般。',
            '奥迪A4L：智能配置多，内饰豪华，保养昂贵。',
            
            # 词向量替换
            '丰田卡罗拉：马力充足，加速有力，油耗经济。',
            '本田思域：外表动感，内装精良，售价略高。',
            '大众速腾：底盘扎实，操控出众，引擎有噪。',
            '丰田卡罗拉：变速平顺，动力线性，节油出色。',
            '本田思域：外形新颖，空间充裕，定价偏贵。',
            '大众速腾：悬挂稳健，转向清晰，机声偏大。',
            '丰田卡罗拉：油电节能，起步静谧，价位偏高。',
            '本田思域：内饰规整，用料讲究，总体满意。',
            '大众速腾：驾控准确，刹车有力，动力中庸。',
            '奥迪A4L：科技丰富，做工精良，养护费用高。'
        ],
        'sentiment_analysis_results': [1, 2, 0] * 10,  # 1:正面, 2:中性, 0:负面
        'aug_method': ['template'] * 10 + ['synonym_dict'] * 10 + ['synonym_pretrain'] * 10
    })
    
    return original_data, augmented_data

def main():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建示例数据
    original_data, augmented_data = create_sample_data()
    logging.info("示例数据创建成功")
    
    # 初始化评估器
    evaluator = QualityEvaluator()
    
    # 按增强方法分组评估
    for method in ['template', 'synonym_dict', 'synonym_pretrain']:
        method_data = augmented_data[augmented_data['aug_method'] == method]
        if len(method_data) == 0:
            logging.info(f"没有找到{method}方法的增强数据")
            continue
            
        logging.info(f"\n评估{method}方法的增强效果...")
        
        # 获取原始文本和增强文本
        original_indices = range(len(method_data))  # 使用示例数据的索引
        original_texts = original_data.loc[original_indices, 'content'].tolist()
        augmented_texts = method_data['content'].tolist()
        original_sentiments = original_data.loc[original_indices, 'sentiment_analysis_results'].tolist()
        
        # 评估
        results = evaluator.evaluate_batch(
            original_texts=original_texts,
            augmented_texts=augmented_texts,
            original_sentiments=original_sentiments,
            method=method
        )
        
        # 打印详细结果
        logging.info(f"样本数量：{results['sample_count']}")
        logging.info(f"语义保持率：{results['semantic_preservation']['rate']:.2%}")
        logging.info(f"情感一致性：{results['sentiment_consistency']['rate']:.2%}")
        logging.info(f"领域相关性：{results['domain_relevance']['rate']:.2%}")
        
    # 创建评估结果目录
    os.makedirs("data/evaluation", exist_ok=True)
    
    # 保存评估结果
    output_file = "data/evaluation/evaluation_results.json"
    evaluator.save_results(output_file)
    logging.info(f"\n评估结果已保存到：{output_file}")
    
    # 打印总体评估报告
    evaluator.print_summary()

if __name__ == "__main__":
    main() 