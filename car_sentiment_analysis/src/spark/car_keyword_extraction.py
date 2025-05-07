# -*- coding: utf-8 -*-
"""
文件名: car_keyword_extraction.py
功能描述: 使用PySpark对汽车评论内容进行关键词提取

主要功能:
1. 数据源处理:
   - 从MySQL数据库读取多个汽车品牌的评论内容
   - 支持批量处理多个数据表
   
2. 文本处理和关键词提取:
   - 使用jieba进行中文分词
   - 过滤停用词
   - 统计词频
   - 按情感分类提取每个车型的TOP 70关键词

3. 结果输出:
   - 将提取的关键词写入MySQL数据库表 'aa_spark_car_keywords'
   - 每个车型产生350条关键词记录（5种情感类型 × 70个关键词）

技术特点:
- 使用PySpark进行大规模文本处理
- jieba中文分词
- 自定义停用词过滤
- 优化的数据库连接和批处理逻辑

作者: [您的名字]
创建日期: 2024-05-03
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, count, desc, row_number, lit
from pyspark.sql.window import Window
import jieba
import os
import logging
import time
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置Python默认编码
import locale
try:
    locale.setlocale(locale.LC_ALL, 'zh_CN.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'Chinese_China.UTF8')
    except:
        pass

def load_stopwords(file_path):
    """
    加载停用词列表
    """
    try:
        stopwords = set()
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and not word.startswith('#'):  # 跳过空行和注释
                    stopwords.add(word)
        return stopwords
    except Exception as e:
        print(f"加载停用词文件时出错: {str(e)}")
        return set()

def setup_logger():
    """
    设置日志记录器
    """
    try:
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        
        log_dir = os.path.join(current_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'keyword_extraction_{current_time}.log')
        
        logger = logging.getLogger('KeywordExtraction')
        logger.setLevel(logging.DEBUG)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    except Exception as e:
        print(f"设置日志记录器时出错: {str(e)}")
        raise

def create_spark_session():
    """
    创建Spark会话
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"当前目录: {current_dir}")
        
        # 设置Hadoop环境变量
        hadoop_home = os.path.join(current_dir, "hadoop")
        os.environ['HADOOP_HOME'] = hadoop_home
        os.environ['HADOOP_BIN_DIR'] = os.path.join(hadoop_home, "bin")
        os.environ['HADOOP_CONF_DIR'] = os.path.join(hadoop_home, "conf")
        os.environ['HADOOP_USER_NAME'] = "root"
        os.environ['JAVA_HOME'] = os.getenv('JAVA_HOME')
        
        # 设置Spark环境
        spark_home = os.path.join(current_dir, "spark_home", "spark-3.5.5-bin-hadoop3")
        os.environ['SPARK_HOME'] = spark_home
        print(f"SPARK_HOME设置为: {spark_home}")
        
        # 添加Spark的Python路径
        spark_python = os.path.join(spark_home, "python")
        py4j_path = os.path.join(spark_python, "lib")
        sys.path.insert(0, spark_python)
        sys.path.insert(0, py4j_path)
        
        # 创建SparkSession，增加内存配置
        print("开始创建SparkSession...")
        spark = SparkSession.builder \
            .appName("CarKeywordExtraction") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.memory.offHeap.enabled", "true") \
            .config("spark.memory.offHeap.size", "4g") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.jars", os.path.join(current_dir, "lib", "mysql-connector-j-8.0.33.jar")) \
            .config("spark.driver.extraClassPath", os.path.join(current_dir, "lib", "mysql-connector-j-8.0.33.jar")) \
            .config("spark.ui.showConsoleProgress", "true") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
            .config("spark.sql.shuffle.partitions", "10") \
            .config("spark.default.parallelism", "10") \
            .getOrCreate()
            
        spark.sparkContext.setLogLevel("INFO")
        
        print("SparkSession创建成功!")
        return spark
    except Exception as e:
        print(f"创建Spark会话时出错: {str(e)}")
        raise

def get_mysql_properties():
    """
    获取MySQL连接属性
    """
    return {
        "driver": "com.mysql.cj.jdbc.Driver",
        "url": "jdbc:mysql://localhost:3306/media_crawler_raw_data_dy",
        "user": "root",
        "password": "root"
    }

def segment_text(text, stopwords):
    """
    对文本进行分词并过滤停用词
    """
    if not isinstance(text, str):
        return []
    
    # 使用jieba进行分词
    words = jieba.cut(text)
    
    # 过滤停用词和空字符
    filtered_words = [word for word in words 
                     if word.strip() 
                     and word not in stopwords 
                     and len(word) > 1  # 过滤单字词
                     and not word.isdigit()  # 过滤纯数字
                     and not all(char.isascii() for char in word)]  # 过滤纯英文
    
    return filtered_words

def extract_keywords_by_sentiment(df, sentiment_type, top_n=70):
    """
    按情感类型提取关键词
    
    Args:
        df: 输入的DataFrame
        sentiment_type: 情感类型(0-4)
        top_n: 提取的关键词数量
    Returns:
        DataFrame: 包含car_id, keyword, sentiment_type的结果集
    """
    try:
        # 根据情感类型筛选数据
        if sentiment_type == 4:  # 全部评论
            # 只选择有效的情感分析结果(0,1,2,3)的评论
            filtered_df = df.filter(col("sentiment_analysis_results").isin([0, 1, 2, 3]))
        else:
            # 按特定情感类型筛选
            filtered_df = df.filter(col("sentiment_analysis_results") == sentiment_type)
        
        # 分词和统计
        keywords_df = filtered_df.selectExpr(
            "car_id",
            "explode(segment_text(content)) as word"
        ).groupBy("car_id", "word").count()
        
        # 为每个car_id选择top n关键词
        window_spec = Window.partitionBy("car_id").orderBy(desc("count"))
        keywords_df = keywords_df.withColumn(
            "rank", row_number().over(window_spec)
        ).filter(f"rank <= {top_n}").select(
            "car_id", 
            "word",
            lit(str(sentiment_type)).alias("sentiment_type")  # 确保sentiment_type为字符串类型
        )
        
        return keywords_df
    except Exception as e:
        logger.error(f"处理情感类型 {sentiment_type} 时发生错误: {str(e)}")
        return None

def extract_keywords():
    """
    主函数：提取关键词并保存结果
    """
    try:
        logger = setup_logger()
        start_time = time.time()
        logger.info("开始关键词提取处理")
        
        # 创建Spark会话
        spark = create_spark_session()
        logger.info("Spark会话创建成功")
        
        # 加载停用词
        current_dir = os.path.dirname(os.path.abspath(__file__))
        stopwords_file = os.path.join(current_dir, "tyc.txt")
        stopwords = load_stopwords(stopwords_file)
        logger.info(f"已加载 {len(stopwords)} 个停用词")
        
        mysql_properties = get_mysql_properties()
        
        # 注册UDF用于分词
        from pyspark.sql.types import ArrayType, StringType
        spark.udf.register("segment_text", 
                          lambda x: segment_text(x, stopwords), 
                          ArrayType(StringType()))
        
        processed_tables = 0
        failed_tables = 0
        
        # 定义要处理的汽车表名列表
        car_tables = [
            "toyota_corolla", "nissan_sylphy", "volvo_s60", "volkswagen_lavida",
            "buick_excelle_gt", "volkswagen_sagitar", "volkswagen_passat",
            "cadillac_ct4", "chery_arrizo_8", "honda_civic", "honda_accord",
            "geely_preface", "geely_emgrand", "hongqi_h5", "changan_eado",
            "geely_xingyue_l", "haval_h5", "tank_300", "byd_song_pro",
            "saic_volkswagen_tiguan_l_pro", "bmw_x3", "audi_q5l",
            "buick_envision", "chevrolet_equinox", "jeep_compass",
            "gac_trumpchi_m8", "buick_gl8", "honda_odyssey", "toyota_sienna",
            "saic_volkswagen_viloran", "byd_han", "gac_aion_aion_s",
            "tesla_model_3", "bmw_i3", "mercedes_benz_eqc", "byd_qin_plus_ev",
            "nio_et5", "xpeng_p7", "xiaomi_su7", "chery_fengyun_a8",
            "byd_song_plus_ev", "gac_aion_aion_y", "tesla_model_y", "nio_es6",
            "xpeng_g6", "voyah_dreamer", "denza_d9", "gac_trumpchi_e9",
            "saic_roewe_imax8_ev", "xpeng_x9"
        ]
        
        # 处理每个汽车表
        for table_name in car_tables:
            table_start_time = time.time()
            try:
                logger.info(f"开始处理表: {table_name}")
                
                # 分批读取数据以减少内存使用
                df = spark.read.jdbc(
                    url=mysql_properties["url"],
                    table=table_name,
                    properties=mysql_properties
                )
                
                # 选择需要的列并确保数据类型正确
                df = df.select(
                    col("car_id"),
                    col("content"),
                    col("sentiment_analysis_results").cast("integer")
                ).filter(col("content").isNotNull())  # 过滤空内容
                
                # 缓存过滤后的DataFrame以提高性能
                df.cache()
                
                # 处理每种情感类型（0-4）并直接写入数据库
                for sentiment_type in range(5):
                    logger.info(f"处理表 {table_name} 的情感类型 {sentiment_type}")
                    
                    sentiment_keywords = extract_keywords_by_sentiment(
                        df, 
                        sentiment_type,
                        top_n=70
                    )
                    
                    if sentiment_keywords is not None:
                        # 直接写入数据库，而不是保存在内存中
                        sentiment_keywords.write.jdbc(
                            url=mysql_properties["url"],
                            table="aa_spark_car_keywords",
                            mode="append",  # 使用append模式而不是overwrite
                            properties=mysql_properties
                        )
                        
                        # 输出当前进度
                        keywords_count = sentiment_keywords.count()
                        logger.info(f"表 {table_name} 情感类型 {sentiment_type} 生成关键词数量: {keywords_count}")
                
                # 释放缓存
                df.unpersist()
                
                processed_tables += 1
                table_time = time.time() - table_start_time
                logger.info(f"成功处理表 {table_name}，耗时 {table_time:.2f} 秒")
                
            except Exception as e:
                logger.error(f"处理表 {table_name} 时发生错误: {str(e)}")
                failed_tables += 1
                continue
        
        # 记录总体统计信息
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"关键词提取处理完成")
        logger.info(f"总耗时: {total_time:.2f} 秒")
        logger.info(f"成功处理表数量: {processed_tables}")
        logger.info(f"失败表数量: {failed_tables}")
        
        spark.stop()
        
    except Exception as e:
        logger.error(f"执行关键词提取时出错: {str(e)}")
        raise

if __name__ == "__main__":
    extract_keywords() 