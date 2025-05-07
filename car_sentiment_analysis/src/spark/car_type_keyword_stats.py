# -*- coding: utf-8 -*-
"""
文件名: car_type_keyword_stats.py
功能描述: 使用PySpark统计不同车型类别的关键词

主要功能:
1. 数据源处理:
   - 从MySQL数据库读取多个汽车品牌的评论数据
   - 读取汽车基础信息表(aa_spark_car_info)获取分类信息
   
2. 文本处理和关键词提取:
   - 按车型类别和动力类型分组
   - 使用jieba进行中文分词
   - 过滤停用词
   - 统计词频
   - 提取关键词

3. 结果输出:
   - 将提取的关键词写入MySQL数据库表 'aa_spark_car_type_keywords'
   - 包含类别描述和关键词

技术特点:
- 使用PySpark进行大规模文本处理
- jieba中文分词
- 自定义停用词过滤
- 多表关联和分组统计
- 优化的数据库连接和批处理逻辑

作者: [您的名字]
创建日期: 2024-05-03
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, split, count, desc, row_number, 
    when, concat_ws, lit, expr, collect_list
)
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
        log_file = os.path.join(log_dir, f'car_type_keywords_{current_time}.log')
        
        logger = logging.getLogger('CarTypeKeywords')
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
    创建Spark会话，增加内存配置
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
            .appName("CarTypeKeywordStats") \
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

def process_car_type_keywords():
    """
    主函数：处理车型类别关键词统计
    """
    try:
        logger = setup_logger()
        start_time = time.time()
        logger.info("开始车型类别关键词统计处理")
        
        # 创建Spark会话
        spark = create_spark_session()
        logger.info("Spark会话创建成功")
        
        # 加载停用词
        current_dir = os.path.dirname(os.path.abspath(__file__))
        stopwords_file = os.path.join(current_dir, "tyc.txt")
        stopwords = load_stopwords(stopwords_file)
        logger.info(f"已加载 {len(stopwords)} 个停用词")
        
        mysql_properties = get_mysql_properties()
        
        # 读取汽车基础信息表
        car_info_df = spark.read.jdbc(
            url=mysql_properties["url"],
            table="aa_spark_car_info",
            properties=mysql_properties
        )
        logger.info("已读取汽车基础信息表")
        
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
        
        # 注册UDF用于分词
        from pyspark.sql.types import ArrayType, StringType
        spark.udf.register("segment_text", 
                          lambda x: segment_text(x, stopwords), 
                          ArrayType(StringType()))
        
        processed_tables = 0
        failed_tables = 0
        
        # 处理每个汽车表
        for table_name in car_tables:
            table_start_time = time.time()
            try:
                logger.info(f"开始处理表: {table_name}")
                
                # 读取源表数据
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
                ).filter(col("content").isNotNull())
                
                # 关联汽车基础信息表
                joined_df = df.join(
                    car_info_df,
                    on="car_id",
                    how="inner"
                )
                
                # 缓存关联后的DataFrame
                joined_df.cache()
                
                # 处理每个车型类别
                for car_type in ["SUV", "MPV", "轿车"]:
                    car_type_df = joined_df.filter(col("car_type") == car_type)
                    
                    # 处理每种情感类型
                    for sentiment_type in range(5):
                        logger.info(f"处理车型类别 {car_type} 的情感类型 {sentiment_type}")
                        
                        keywords_df = extract_keywords_by_category_and_sentiment(
                            car_type_df,
                            car_type,
                            sentiment_type,
                            top_n=70
                        )
                        
                        if keywords_df is not None:
                            # 直接写入数据库
                            keywords_df.write.jdbc(
                                url=mysql_properties["url"],
                                table="aa_spark_car_type_keywords",
                                mode="append",
                                properties=mysql_properties
                            )
                
                # 处理动力类型类别
                power_type_map = {"燃油": "燃油", "电动": "新能源"}
                for power_type, category in power_type_map.items():
                    power_type_df = joined_df.filter(col("power_type") == power_type)
                    
                    # 处理每种情感类型
                    for sentiment_type in range(5):
                        logger.info(f"处理动力类型 {category} 的情感类型 {sentiment_type}")
                        
                        keywords_df = extract_keywords_by_category_and_sentiment(
                            power_type_df,
                            category,
                            sentiment_type,
                            top_n=70
                        )
                        
                        if keywords_df is not None:
                            # 直接写入数据库
                            keywords_df.write.jdbc(
                                url=mysql_properties["url"],
                                table="aa_spark_car_type_keywords",
                                mode="append",
                                properties=mysql_properties
                            )
                
                # 释放缓存
                joined_df.unpersist()
                
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
        logger.info(f"车型类别关键词统计处理完成")
        logger.info(f"总耗时: {total_time:.2f} 秒")
        logger.info(f"成功处理表数量: {processed_tables}")
        logger.info(f"失败表数量: {failed_tables}")
        
        spark.stop()
        
    except Exception as e:
        logger.error(f"执行关键词统计时出错: {str(e)}")
        raise

def extract_keywords_by_category_and_sentiment(df, category_description, sentiment_type, top_n=70):
    """
    按类别和情感类型提取关键词
    
    Args:
        df: 输入的DataFrame
        category_description: 类别描述
        sentiment_type: 情感类型(0-4)
        top_n: 提取的关键词数量
    Returns:
        DataFrame: 包含category_description, sentiment_type, keyword的结果集
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
            f"'{category_description}' as category_description",
            "explode(segment_text(content)) as word"
        ).groupBy("category_description", "word").count()
        
        # 为每个类别选择top n关键词
        window_spec = Window.partitionBy("category_description").orderBy(desc("count"))
        keywords_df = keywords_df.withColumn(
            "rank", row_number().over(window_spec)
        ).filter(f"rank <= {top_n}").select(
            "category_description",
            lit(str(sentiment_type)).alias("sentiment_type"),
            col("word").alias("keyword")
        )
        
        return keywords_df
    except Exception as e:
        logger.error(f"处理类别 {category_description} 情感类型 {sentiment_type} 时发生错误: {str(e)}")
        return None

if __name__ == "__main__":
    process_car_type_keywords() 