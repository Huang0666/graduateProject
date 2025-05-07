"""
批量处理数据库中的评论数据进行情感分析
"""

import os
import sys
import logging
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from tqdm import tqdm
import time
import argparse
import json

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

# 导入模型相关代码
from car_sentiment_analysis.src.model_process.model_train import CarSentimentModel

# 配置日志
log_file = f'sentiment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # 添加utf-8编码
        logging.StreamHandler()
    ]
)

# 输出启动信息
logging.info("="*50)
logging.info("情感分析处理程序启动")
logging.info("="*50)

# 模型配置常量
REQUIRED_CONFIG = {
    'model_name': 'bert-base-chinese',
    'num_labels': 4,
    'max_length': 128
}

# 模型版本配置
MODEL_VERSIONS = {
    'v3': {
        'path': 'src/saved_models/checkpoints/v3/best_model.pth',
        'description': '6000条数据训练的四分类模型 (0:负面, 1:正面, 2:中性, 3:无关)',
        'config': REQUIRED_CONFIG
    },
    'v4': {
        'path': 'src/saved_models/checkpoints/v4/best_model.pth',
        'description': '30000条数据训练的四分类模型 (0:负面, 1:正面, 2:中性, 3:无关)',
        'config': REQUIRED_CONFIG
    }
}

# 车型表名列表
CAR_TABLES = [
    'toyota_corolla', 'nissan_sylphy', 'volvo_s60', 'volkswagen_lavida',
    'buick_excelle_gt', 'volkswagen_sagitar', 'volkswagen_passat',
    'cadillac_ct4', 'chery_arrizo_8', 'honda_civic', 'honda_accord',
    'geely_preface', 'geely_emgrand', 'hongqi_h5', 'changan_eado',
    'geely_xingyue_l', 'haval_h5', 'tank_300', 'byd_song_pro',
    'saic_volkswagen_tiguan_l_pro', 'bmw_x3', 'audi_q5l', 'buick_envision',
    'chevrolet_equinox', 'jeep_compass', 'gac_trumpchi_m8', 'buick_gl8',
    'honda_odyssey', 'toyota_sienna', 'saic_volkswagen_viloran', 'byd_han',
    'gac_aion_aion_s', 'tesla_model_3', 'bmw_i3', 'mercedes_benz_eqc',
    'byd_qin_plus_ev', 'nio_et5', 'xpeng_p7', 'xiaomi_su7'
     'chery_fengyun_a8',
    'byd_song_plus_ev', 'gac_aion_aion_y', 'tesla_model_y', 'nio_es6',
    'xpeng_g6', 'voyah_dreamer', 'denza_d9', 'gac_trumpchi_e9',
    'saic_roewe_imax8_ev', 'xpeng_x9'
]

car_names = [
    "丰田卡罗拉", "日产轩逸", "沃尔沃S60", "大众朗逸", "别克英朗",
    "速腾", "帕萨特", "凯迪拉克 CT4", "艾瑞泽8", "思域",
    "雅阁", "吉利星瑞", "吉利帝豪", "红旗H5", "长安逸动",
    "星越 L", "哈弗 H5", "坦克 300", "比亚迪宋 Pro", "上汽大众途观 L Pro",
    "宝马 X3", "奥迪 Q5L", "别克昂科威", "雪佛兰探界者", "JEEP 指南者",
    "传祺 M8", "别克 GL8", "本田奥德赛", "丰田赛那", "上汽大众威然",
    "比亚迪汉", "广汽埃安 AION S", "特斯拉 Model 3", "宝马 i3", "奔驰 EQC",
    "比亚迪秦 PLUS EV", "蔚来 ET5", "小鹏 P7", "小米su7"
    "奇瑞风云A8", "比亚迪宋 PLUS EV", "广汽埃安 AION Y", "特斯拉 Model Y",
    "蔚来 ES6", "小鹏 G6", "岚图梦想家", "腾势 D9", "广汽传祺 E9",
    "上汽荣威 iMAX8 EV", "小鹏 X9"
]

def validate_model_config(checkpoint_config):
    """验证模型配置是否符合要求"""
    for key, value in REQUIRED_CONFIG.items():
        if checkpoint_config.get(key) != value:
            raise ValueError(f"模型配置错误: {key} 应该为 {value}，"
                           f"实际为 {checkpoint_config.get(key)}")

def validate_input_data(texts, likes, replies):
    """验证输入数据的有效性"""
    if not texts or len(texts) == 0:
        raise ValueError("输入文本不能为空")
    
    if len(likes) != len(texts) or len(replies) != len(texts):
        raise ValueError("点赞数和回复数的数量必须与文本数量相同")
    
    # 检查空值
    empty_texts = [i for i, t in enumerate(texts) if pd.isna(t) or str(t).strip() == '']
    if empty_texts:
        raise ValueError(f"发现空文本，索引位置：{empty_texts}")

class CommentDataset(Dataset):
    """评论数据集类"""
    def __init__(self, texts, likes, replies, tokenizer, max_length=128):
        # 验证输入数据
        validate_input_data(texts, likes, replies)
        
        self.texts = texts
        self.likes = likes
        self.replies = replies
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 确保文本不为空且为字符串类型
        text = str(self.texts[idx]).strip()
        if not text:
            raise ValueError(f"索引 {idx} 处的文本为空")
        
        # 处理VARCHAR类型的点赞数和回复数
        try:
            # 处理点赞数：移除非数字字符，转换为浮点数
            like_str = str(self.likes[idx]).strip()
            like_count = float(''.join(c for c in like_str if c.isdigit() or c == '.') or '0')
        except (ValueError, TypeError):
            logging.warning(f"索引 {idx} 处的点赞数无效 '{self.likes[idx]}'，使用默认值0")
            like_count = 0.0
            
        try:
            # 处理回复数：移除非数字字符，转换为浮点数
            reply_str = str(self.replies[idx]).strip()
            reply_count = float(''.join(c for c in reply_str if c.isdigit() or c == '.') or '0')
        except (ValueError, TypeError):
            logging.warning(f"索引 {idx} 处的回复数无效 '{self.replies[idx]}'，使用默认值0")
            reply_count = 0.0

        # 文本编码
        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        except Exception as e:
            raise ValueError(f"文本编码错误，索引 {idx}：{str(e)}")

        # 社交特征处理
        social_features = torch.tensor(
            [like_count, reply_count],
            dtype=torch.float32
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'social_features': social_features
        }

def load_threshold_config(config_path='threshold_config.json'):
    """加载阈值配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            threshold = config.get('best_threshold', 0.5)
            logging.info(f"已加载阈值配置：{threshold}")
            return threshold
    except Exception as e:
        logging.warning(f"加载阈值配置失败：{e}，使用默认阈值0.5")
        return 0.5

class SentimentAnalyzer:
    """情感分析器类"""
    def __init__(self, model_path, batch_size=32, threshold=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        
        # 加载阈值
        self.threshold = threshold if threshold is not None else load_threshold_config()
        logging.info(f"使用阈值：{self.threshold}")
        
        # 加载模型
        logging.info(f"正在加载模型: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.config = checkpoint['config']
            
            # 验证模型配置
            validate_model_config(self.config)
            
            # 初始化模型
            self.model = CarSentimentModel(
                model_name=self.config['model_name'],
                num_labels=self.config['num_labels']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 初始化tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(self.config['model_name'])
            
            logging.info("模型加载完成")
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def analyze_batch(self, texts, likes, replies):
        """批量分析评论情感"""
        try:
            # 创建数据集
            dataset = CommentDataset(texts, likes, replies, self.tokenizer, 
                                   max_length=self.config['max_length'])
            dataloader = DataLoader(dataset, batch_size=self.batch_size)
            
            predictions = []
            
            # 使用tqdm显示处理进度
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="处理评论批次"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    social_features = batch['social_features'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask, social_features)
                    probs = torch.softmax(outputs, dim=1)
                    
                    # 应用阈值策略
                    positive_probs = probs[:, 1]  # 正向概率
                    positive_mask = positive_probs > self.threshold
                    
                    # 在其他类别中选择最高概率
                    other_probs = probs.clone()
                    other_probs[:, 1] = float('-inf')  # 将正向概率设为负无穷
                    other_preds = other_probs.argmax(dim=1)
                    
                    # 使用掩码合并预测结果
                    batch_preds = torch.where(positive_mask, 
                                           torch.ones_like(other_preds), 
                                           other_preds)
                    
                    # 将预测结果转移到CPU并转换为Python列表
                    predictions.extend(batch_preds.cpu().tolist())
            
            return predictions
            
        except Exception as e:
            logging.error(f"批处理过程出错: {str(e)}")
            raise

class DatabaseHandler:
    """数据库处理器类"""
    def __init__(self):
        # 加载环境变量
        load_dotenv()
        
        # 数据库配置
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', 'your_password'),
            'database': os.getenv('DB_NAME', 'media_crawler_raw_data_dy'),
            'charset': 'utf8mb4'
        }
        
        # 验证数据库配置
        self._validate_config()

    def _validate_config(self):
        """验证数据库配置"""
        required_fields = ['host', 'port', 'user', 'password', 'database']
        missing_fields = [field for field in required_fields 
                         if not self.db_config.get(field)]
        if missing_fields:
            raise ValueError(f"数据库配置缺少必要字段: {', '.join(missing_fields)}")

    def connect(self):
        """连接到数据库"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            return connection
        except Error as e:
            logging.error(f"数据库连接错误: {e}")
            raise

    def get_unprocessed_comments(self, table_name, batch_size=1000):
        """获取未处理的评论数据"""
        connection = self.connect()
        cursor = connection.cursor()
        
        try:
            # 验证表名
            if table_name not in CAR_TABLES:
                raise ValueError(f"无效的表名: {table_name}")
            
            # 获取对应的中文车名
            car_name = car_names[CAR_TABLES.index(table_name)]
            
            query = f"""
                SELECT id, content, like_count, sub_comment_count 
                FROM {table_name}
                WHERE sentiment_analysis_results IS NULL
                LIMIT {batch_size}
            """
            cursor.execute(query)
            raw_results = cursor.fetchall()
            
            if not raw_results:
                logging.info(f"表 {table_name} 中没有未处理的数据")
                return []
            
            # 在评论内容前添加中文车名
            results = []
            for result in raw_results:
                id_, content, like_count, sub_comment_count = result
                modified_content = f"{car_name}：{content}"
                results.append((id_, modified_content, like_count, sub_comment_count))
            
            return results
            
        except Error as e:
            logging.error(f"查询错误 {table_name}: {e}")
            return []
        finally:
            cursor.close()
            connection.close()

    def update_sentiment_results(self, table_name, id_results):
        """更新情感分析结果"""
        if not id_results:
            logging.warning("没有结果需要更新")
            return True
            
        connection = self.connect()
        cursor = connection.cursor()
        
        try:
            # 验证表名
            if table_name not in CAR_TABLES:
                raise ValueError(f"无效的表名: {table_name}")
            
            # 转换numpy.int64为Python原生int类型
            converted_results = [(int(pred), int(id_)) for pred, id_ in id_results]
            
            query = f"""
                UPDATE {table_name}
                SET sentiment_analysis_results = %s
                WHERE id = %s
            """
            cursor.executemany(query, converted_results)
            connection.commit()
            return True
            
        except Error as e:
            logging.error(f"更新错误 {table_name}: {e}")
            connection.rollback()
            return False
        finally:
            cursor.close()
            connection.close()

def main():
    """主函数"""
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='汽车评论情感分析处理程序')
        parser.add_argument('--model_version', type=str, choices=['v3', 'v4'],
                          default='v3', help='选择模型版本：v3(6000条数据) 或 v4(30000条数据)')
        parser.add_argument('--batch_size', type=int, default=32,
                          help='模型预测的批处理大小')
        parser.add_argument('--max_retries', type=int, default=3,
                          help='处理失败时的最大重试次数')
        args = parser.parse_args()
        
        # 获取选择的模型配置
        model_config = MODEL_VERSIONS[args.model_version]
        model_path = os.path.join(project_root, model_config['path'])
        
        # 验证模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 输出模型信息
        logging.info(f"使用模型版本: {args.model_version}")
        logging.info(f"模型说明: {model_config['description']}")
        logging.info(f"模型路径: {model_path}")
        
        # 初始化情感分析器
        analyzer = SentimentAnalyzer(model_path, batch_size=args.batch_size)
        db_handler = DatabaseHandler()
        
        # 处理每个车型表
        total_tables = len(CAR_TABLES)
        for idx, table_name in enumerate(CAR_TABLES, 1):
            logging.info(f"\n处理进度: [{idx}/{total_tables}] 表: {table_name}")
            
            while True:
                # 获取一批未处理的数据
                results = db_handler.get_unprocessed_comments(table_name)
                if not results:
                    logging.info(f"表 {table_name} 处理完成")
                    break
                
                # 准备数据
                ids = [int(r[0]) for r in results]
                texts = [r[1] for r in results]
                likes = [r[2] for r in results]
                replies = [r[3] for r in results]
                
                # 进行情感分析（带重试机制）
                retry_count = 0
                success = False
                last_error = None
                
                while retry_count < args.max_retries and not success:
                    try:
                        if retry_count > 0:
                            logging.info(f"第 {retry_count} 次重试...")
                            
                        predictions = analyzer.analyze_batch(texts, likes, replies)
                        
                        # 准备更新数据
                        id_results = list(zip(predictions, ids))
                        
                        # 更新数据库
                        if db_handler.update_sentiment_results(table_name, id_results):
                            logging.info(f"成功更新 {len(id_results)} 条记录")
                            success = True
                        else:
                            raise Exception("数据库更新失败")
                            
                    except Exception as e:
                        last_error = e
                        retry_count += 1
                        if retry_count >= args.max_retries:
                            logging.error(f"处理失败 {table_name}: {str(e)}")
                            logging.error(f"已达到最大重试次数 {args.max_retries}，跳过当前批次")
                        time.sleep(1)  # 重试前等待1秒
                
                if not success:
                    logging.error(f"处理错误 {table_name}: {str(last_error)}")
                    # 记录失败的数据ID
                    with open(f'failed_records_{datetime.now().strftime("%Y%m%d")}.log', 'a', encoding='utf-8') as f:
                        f.write(f"\n表名: {table_name}\n")
                        f.write(f"失败原因: {str(last_error)}\n")
                        f.write(f"ID列表: {','.join(map(str, ids))}\n")
                        f.write("-" * 50 + "\n")
                
                # 成功或达到最大重试次数后，继续处理下一批
                time.sleep(0.1)
    
    except Exception as e:
        logging.error(f"主程序执行错误: {e}")
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"程序执行错误: {e}")
        raise 