# -*- coding: utf-8 -*-
"""
文件名: car_type_interaction_stats.py
功能描述: 使用PySpark统计不同车型类别的评论互动数据

主要功能:
1. 数据源处理:
   - 从MySQL数据库读取多个汽车品牌的评论数据
   - 读取汽车基础信息表(aa_spark_car_info)获取分类信息
   
2. 数据统计:
   - 按车型类别(SUV/MPV/轿车)统计
   - 按动力类型(燃油/新能源)统计
   - 计算点赞量、评论量和总互动量
   - 每个类别选择TOP 50的评论数据

3. 结果输出:
   - 将统计结果写入MySQL数据库表 'aa_spark_car_type_comments'
   - 包含类别、评论内容、互动数据和互动类型

技术特点:
- 使用PySpark进行大规模数据处理
- 多表关联查询
- 分组统计和排序
- 优化的数据库连接和批处理逻辑

作者: [您的名字]
创建日期: 2024-05-03
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, sum, desc, row_number, when, 
    concat_ws, lit, expr
)
from pyspark.sql.window import Window
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
    """
    设置日志记录器
    """
    try:
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        
        log_dir = os.path.join(current_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'car_type_stats_{current_time}.log')
        
        logger = logging.getLogger('CarTypeStats')
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
        
        # 创建SparkSession
        print("开始创建SparkSession...")
        spark = SparkSession.builder \
            .appName("CarTypeInteractionStats") \
            .master("local[*]") \
            .config("spark.jars", os.path.join(current_dir, "lib", "mysql-connector-j-8.0.33.jar")) \
            .config("spark.driver.extraClassPath", os.path.join(current_dir, "lib", "mysql-connector-j-8.0.33.jar")) \
            .config("spark.ui.showConsoleProgress", "true") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
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

def process_car_type_stats():
    """
    主函数：处理车型类别统计
    """
    try:
        logger = setup_logger()
        start_time = time.time()
        logger.info("开始车型类别统计处理")
        
        # 创建Spark会话
        logger.info("正在创建Spark会话...")
        spark = create_spark_session()
        logger.info("Spark会话创建成功")
        
        mysql_properties = get_mysql_properties()
        logger.info(f"MySQL连接配置: {mysql_properties}")
        
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
        
        # 用于存储所有评论数据的DataFrame
        all_comments_df = None
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
                
                # 选择需要的列
                df = df.select(
                    "car_id",
                    "content",
                    col("like_count").alias("digg_count"),
                    col("sub_comment_count").alias("comment_count")
                )
                
                # 合并结果
                if all_comments_df is None:
                    all_comments_df = df
                else:
                    all_comments_df = all_comments_df.union(df)
                
                # 记录处理时间
                table_time = time.time() - table_start_time
                logger.info(f"成功处理表 {table_name}，耗时 {table_time:.2f} 秒")
                processed_tables += 1
                
            except Exception as e:
                logger.error(f"处理表 {table_name} 时发生错误: {str(e)}")
                failed_tables += 1
                continue
        
        if all_comments_df is not None:
            try:
                # 关联汽车基础信息表
                joined_df = all_comments_df.join(
                    car_info_df,
                    on="car_id",
                    how="inner"
                )
                
                # 计算总互动量
                joined_df = joined_df.withColumn(
                    "total_interaction",
                    col("digg_count") + col("comment_count")
                )
                
                # 处理车型类别统计
                car_type_results = []
                
                # 处理SUV/MPV/轿车类别
                for car_type in ["SUV", "MPV", "轿车"]:
                    # 点赞量TOP 50
                    window_spec = Window.partitionBy().orderBy(desc("digg_count"))
                    likes_df = joined_df.filter(col("car_type") == car_type) \
                        .withColumn("rank", row_number().over(window_spec)) \
                        .filter(col("rank") <= 50) \
                        .select(
                            lit(car_type).alias("category_description"),
                            "content",
                            col("digg_count").cast("string").alias("interaction_count"),
                            lit("0").alias("interaction_type")
                        )
                    car_type_results.append(likes_df)
                    
                    # 评论量TOP 50
                    window_spec = Window.partitionBy().orderBy(desc("comment_count"))
                    comments_df = joined_df.filter(col("car_type") == car_type) \
                        .withColumn("rank", row_number().over(window_spec)) \
                        .filter(col("rank") <= 50) \
                        .select(
                            lit(car_type).alias("category_description"),
                            "content",
                            col("comment_count").cast("string").alias("interaction_count"),
                            lit("1").alias("interaction_type")
                        )
                    car_type_results.append(comments_df)
                    
                    # 总互动量TOP 50
                    window_spec = Window.partitionBy().orderBy(desc("total_interaction"))
                    total_df = joined_df.filter(col("car_type") == car_type) \
                        .withColumn("rank", row_number().over(window_spec)) \
                        .filter(col("rank") <= 50) \
                        .select(
                            lit(car_type).alias("category_description"),
                            "content",
                            col("total_interaction").cast("string").alias("interaction_count"),
                            lit("2").alias("interaction_type")
                        )
                    car_type_results.append(total_df)
                
                # 处理燃油/新能源类别
                power_type_map = {"燃油": "燃油", "电动": "新能源"}
                for power_type, category in power_type_map.items():
                    # 点赞量TOP 50
                    window_spec = Window.partitionBy().orderBy(desc("digg_count"))
                    likes_df = joined_df.filter(col("power_type") == power_type) \
                        .withColumn("rank", row_number().over(window_spec)) \
                        .filter(col("rank") <= 50) \
                        .select(
                            lit(category).alias("category_description"),
                            "content",
                            col("digg_count").cast("string").alias("interaction_count"),
                            lit("0").alias("interaction_type")
                        )
                    car_type_results.append(likes_df)
                    
                    # 评论量TOP 50
                    window_spec = Window.partitionBy().orderBy(desc("comment_count"))
                    comments_df = joined_df.filter(col("power_type") == power_type) \
                        .withColumn("rank", row_number().over(window_spec)) \
                        .filter(col("rank") <= 50) \
                        .select(
                            lit(category).alias("category_description"),
                            "content",
                            col("comment_count").cast("string").alias("interaction_count"),
                            lit("1").alias("interaction_type")
                        )
                    car_type_results.append(comments_df)
                    
                    # 总互动量TOP 50
                    window_spec = Window.partitionBy().orderBy(desc("total_interaction"))
                    total_df = joined_df.filter(col("power_type") == power_type) \
                        .withColumn("rank", row_number().over(window_spec)) \
                        .filter(col("rank") <= 50) \
                        .select(
                            lit(category).alias("category_description"),
                            "content",
                            col("total_interaction").cast("string").alias("interaction_count"),
                            lit("2").alias("interaction_type")
                        )
                    car_type_results.append(total_df)
                
                # 合并所有结果
                final_df = car_type_results[0]
                for df in car_type_results[1:]:
                    final_df = final_df.union(df)
                
                # 将结果写入目标表
                final_df.write.jdbc(
                    url=mysql_properties["url"],
                    table="aa_spark_car_type_comments",
                    mode="overwrite",
                    properties=mysql_properties
                )
                logger.info("成功将统计结果写入目标表")
                
            except Exception as e:
                logger.error(f"处理统计结果时发生错误: {str(e)}")
        else:
            logger.warning("没有成功处理任何数据表")
        
        # 记录总体统计信息
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"车型类别统计处理完成")
        logger.info(f"总耗时: {total_time:.2f} 秒")
        logger.info(f"成功处理表数量: {processed_tables}")
        logger.info(f"失败表数量: {failed_tables}")
        
        spark.stop()
        
    except Exception as e:
        logger.error(f"执行统计处理时出错: {str(e)}")
        raise

if __name__ == "__main__":
    process_car_type_stats() 