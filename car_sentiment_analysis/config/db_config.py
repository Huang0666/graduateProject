# 数据库配置
# 使用环境变量或默认值来配置数据库连接
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 数据库配置
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'your_password'),
    'database': os.getenv('DB_NAME', 'car_sentiment'),
    'charset': 'utf8mb4',
    'table': os.getenv('DB_TABLE', 'car_comments'),  # 评论表名
    'text_column': os.getenv('TEXT_COLUMN', 'comment_text'),  # 评论文本列名
    'id_column': os.getenv('ID_COLUMN', 'id'),  # 主键ID列名
    'sentiment_column': os.getenv('SENTIMENT_COLUMN', 'sentiment'),  # 情感标签列名
}

# 源数据库配置（用于批量处理）
SOURCE_DB_CONFIG = {
    'host': os.getenv('SOURCE_DB_HOST', DB_CONFIG['host']),
    'port': int(os.getenv('SOURCE_DB_PORT', DB_CONFIG['port'])),
    'user': os.getenv('SOURCE_DB_USER', DB_CONFIG['user']),
    'password': os.getenv('SOURCE_DB_PASSWORD', DB_CONFIG['password']),
    'database': os.getenv('SOURCE_DB_NAME', 'media_crawler_raw_data_dy'),  # 源数据库名
    'charset': 'utf8mb4'
} 