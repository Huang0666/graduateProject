# -*- coding: utf-8 -*-
"""
文件名: car_sentiment_aggregation.py
功能描述: 使用PySpark对汽车评论情感分析结果进行数据聚合处理

主要功能:
1. 数据源处理:
   - 从MySQL数据库读取多个汽车品牌的评论情感分析结果
   - 支持批量处理多个数据表
   
2. 数据聚合统计:
   - 统计每个汽车品牌的正面、负面、中性和无关评论数量
   - 聚合计算包括：
     * 正面评价数量 (sentiment_analysis_results = 1)
     * 负面评价数量 (sentiment_analysis_results = 0)
     * 中性评价数量 (sentiment_analysis_results = 2)
     * 无关评价数量 (sentiment_analysis_results = 3)

3. 结果输出:
   - 将聚合结果写入MySQL数据库表 'aa_spark_car_evaluation_stats'
   - 包含每个汽车的统计数据和评价分布

技术特点:
- 使用PySpark进行大规模数据处理
- 支持Windows环境下的Spark配置
- 实现日志记录和错误处理
- 优化的数据库连接和批处理逻辑

作者: [您的名字]
创建日期: 2024-05-03
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, first
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

def setup_logger():
    try:
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        
        log_dir = os.path.join(current_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'spark_aggregation_{current_time}.log')
        
        logger = logging.getLogger('SparkAggregation')
        logger.setLevel(logging.DEBUG)
        
        # 文件处理器使用utf-8编码
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        
        # 设置日志格式
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
    try:
        # 获取当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"当前目录: {current_dir}")
        
        # 设置Hadoop环境变量
        hadoop_home = os.path.join(current_dir, "hadoop")
        os.environ['HADOOP_HOME'] = hadoop_home
        os.environ['HADOOP_BIN_DIR'] = os.path.join(hadoop_home, "bin")
        os.environ['HADOOP_CONF_DIR'] = os.path.join(hadoop_home, "conf")
        os.environ['HADOOP_USER_NAME'] = "root"
        os.environ['JAVA_HOME'] = os.getenv('JAVA_HOME')  # 确保JAVA_HOME已设置
        
        # 设置Spark环境
        spark_home = os.path.join(current_dir, "spark_home", "spark-3.5.5-bin-hadoop3")
        os.environ['SPARK_HOME'] = spark_home
        print(f"SPARK_HOME设置为: {spark_home}")
        
        # 添加Spark的Python路径
        spark_python = os.path.join(spark_home, "python")
        py4j_path = os.path.join(spark_python, "lib")
        sys.path.insert(0, spark_python)
        sys.path.insert(0, py4j_path)
        print(f"Python路径已更新")
        
        # 创建SparkSession
        print("开始创建SparkSession...")
        spark = SparkSession.builder \
            .appName("CarSentimentAggregation") \
            .master("local[*]") \
            .config("spark.jars", os.path.join(current_dir, "lib", "mysql-connector-j-8.0.33.jar")) \
            .config("spark.driver.extraClassPath", os.path.join(current_dir, "lib", "mysql-connector-j-8.0.33.jar")) \
            .config("spark.ui.showConsoleProgress", "true") \
            .config("spark.hadoop.fs.defaultFS", "file:///") \
            .getOrCreate()
            
        # 设置日志级别
        spark.sparkContext.setLogLevel("INFO")
        
        print("SparkSession创建成功!")
        return spark
    except Exception as e:
        print(f"创建Spark会话时出错: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            print("完整错误堆栈:")
            traceback.print_exc()
        raise

def get_mysql_properties():
    return {
        "driver": "com.mysql.cj.jdbc.Driver",
        "url": "jdbc:mysql://localhost:3306/media_crawler_raw_data_dy",
        "user": "root",
        "password": "root"
    }

def aggregate_car_sentiments():
    try:
        logger = setup_logger()
        start_time = time.time()
        logger.info("开始数据聚合处理")
        
        # 创建Spark会话
        logger.info("正在创建Spark会话...")
        spark = create_spark_session()
        logger.info("Spark会话创建成功")
        
        mysql_properties = get_mysql_properties()
        logger.info(f"MySQL连接配置: {mysql_properties}")
        
        # 定义要处理的汽车表名列表
        car_tables = [
            "toyota_corolla",   
            "nissan_sylphy",
            "volvo_s60",
            "volkswagen_lavida",
            "buick_excelle_gt",
            "volkswagen_sagitar",
            "volkswagen_passat",
            "cadillac_ct4",
            "chery_arrizo_8",
            "honda_civic",
            "honda_accord",
            "geely_preface",
            "geely_emgrand",
            "hongqi_h5",
            "changan_eado",
            "geely_xingyue_l",
            "haval_h5",
            "tank_300",
            "byd_song_pro",
            "saic_volkswagen_tiguan_l_pro",
            "bmw_x3",
            "audi_q5l",
            "buick_envision",
            "chevrolet_equinox",
            "jeep_compass",
            "gac_trumpchi_m8",
            "buick_gl8",
            "honda_odyssey",
            "toyota_sienna",
            "saic_volkswagen_viloran",
            "byd_han",
            "gac_aion_aion_s",
            "tesla_model_3",
            "bmw_i3",
            "mercedes_benz_eqc",
            "byd_qin_plus_ev",
            "nio_et5",
            "xpeng_p7",
            "xiaomi_su7",
            "chery_fengyun_a8",
            "byd_song_plus_ev",
            "gac_aion_aion_y",
            "tesla_model_y",
            "nio_es6",
            "xpeng_g6",
            "voyah_dreamer",
            "denza_d9",
            "gac_trumpchi_e9",
            "saic_roewe_imax8_ev",
            "xpeng_x9"
        ]
        
        # 创建空的结果列表
        results = []
        processed_tables = 0
        failed_tables = 0
        
        # 处理每个汽车表
        for table_name in car_tables:
            table_start_time = time.time()
            try:
                # 读取源表数据
                df = spark.read.jdbc(
                    url=mysql_properties["url"],
                    table=table_name,
                    properties=mysql_properties
                )
                
                # 获取数据行数
                row_count = df.count()
                logger.info(f"表 {table_name} 包含 {row_count} 条记录")
                
                # 获取car_id并统计各种情感分析结果的数量
                stats = df.agg(
                    first("car_id").alias("car_id"),
                    count(when(col("sentiment_analysis_results") == 1, 1)).alias("positive_evaluation_count"),
                    count(when(col("sentiment_analysis_results") == 0, 1)).alias("negative_evaluation_count"),
                    count(when(col("sentiment_analysis_results") == 2, 1)).alias("neutral_evaluation_count"),
                    count(when(col("sentiment_analysis_results") == 3, 1)).alias("irrelevant_evaluation_count")
                ).collect()[0]
                
                # 将结果添加到列表中
                results.append({
                    "car_id": stats["car_id"],
                    "positive_evaluation_count": int(stats["positive_evaluation_count"]),
                    "negative_evaluation_count": int(stats["negative_evaluation_count"]),
                    "neutral_evaluation_count": int(stats["neutral_evaluation_count"]),
                    "irrelevant_evaluation_count": int(stats["irrelevant_evaluation_count"])
                })
                
                # 记录处理时间和结果
                table_time = time.time() - table_start_time
                logger.info(f"成功处理表 {table_name}，耗时 {table_time:.2f} 秒")
                logger.info(f"统计结果 - 正面: {stats['positive_evaluation_count']}, 负面: {stats['negative_evaluation_count']}, " 
                           f"中性: {stats['neutral_evaluation_count']}, 无关: {stats['irrelevant_evaluation_count']}")
                processed_tables += 1
                
            except Exception as e:
                logger.error(f"处理表 {table_name} 时发生错误: {str(e)}")
                failed_tables += 1
                continue
        
        if results:
            try:
                # 创建结果DataFrame
                result_df = spark.createDataFrame(results)
                
                # 将结果写入目标表
                result_df.write \
                    .jdbc(
                        url=mysql_properties["url"],
                        table="aa_spark_car_evaluation_stats",
                        mode="overwrite",
                        properties=mysql_properties
                    )
                logger.info("成功将统计结果写入目标表")
            except Exception as e:
                logger.error(f"写入结果到目标表时发生错误: {str(e)}")
        else:
            logger.warning("没有成功处理任何数据表")
        
        # 记录总体统计信息
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"数据聚合处理完成")
        logger.info(f"总耗时: {total_time:.2f} 秒")
        logger.info(f"成功处理表数量: {processed_tables}")
        logger.info(f"失败表数量: {failed_tables}")
        
        spark.stop()
    except Exception as e:
        logger.error(f"执行聚合操作时出错: {str(e)}")
        raise

if __name__ == "__main__":
    aggregate_car_sentiments() 