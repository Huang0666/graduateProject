# -*- coding: utf-8 -*-
"""
文件名: car_comment_aggregation.py
功能描述: 使用PySpark对汽车评论数据进行聚合分析，筛选高互动评论

主要功能:
1. 数据源处理:
   - 从MySQL数据库读取多个汽车品牌的评论数据
   - 每个车型筛选150条高价值评论
   
2. 评论筛选规则:
   - 点赞量最高的50条评论 (interaction_type = '0')
   - 评论量最高的50条评论 (interaction_type = '1')
   - 总互动量最高的50条评论 (interaction_type = '2')

3. 结果输出:
   - 将筛选结果写入MySQL数据库表 'aa_spark_car_comments'
   - 保持原始数据类型，评论内容限制在255字符

技术特点:
- 使用PySpark进行大规模数据处理
- 支持Windows环境下的Spark配置
- 实现日志记录和错误处理
- 优化的数据库连接和批处理逻辑

创建日期: 2024-05-03
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring, concat_ws, desc, row_number, lit
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
    try:
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        
        log_dir = os.path.join(current_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'comment_aggregation_{current_time}.log')
        
        logger = logging.getLogger('CommentAggregation')
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
        print(f"Python路径已更新")
        
        # 创建SparkSession
        print("开始创建SparkSession...")
        spark = SparkSession.builder \
            .appName("CarCommentAggregation") \
            .master("local[*]") \
            .config("spark.jars", os.path.join(current_dir, "lib", "mysql-connector-j-8.0.33.jar")) \
            .config("spark.driver.extraClassPath", os.path.join(current_dir, "lib", "mysql-connector-j-8.0.33.jar")) \
            .config("spark.ui.showConsoleProgress", "true") \
            .config("spark.hadoop.fs.defaultFS", "file:///") \
            .getOrCreate()
            
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

def process_table_comments(spark, table_name, mysql_properties, logger):
    try:
        # 读取源表数据
        df = spark.read.jdbc(
            url=mysql_properties["url"],
            table=table_name,
            properties=mysql_properties
        )
        
        # 截断评论内容到255字符
        df = df.withColumn("comment", substring(col("content"), 1, 255))
        
        # 创建窗口函数，按car_id分组
        window_likes = Window.partitionBy("car_id").orderBy(desc("like_count"))
        window_comments = Window.partitionBy("car_id").orderBy(desc("sub_comment_count"))
        window_total = Window.partitionBy("car_id").orderBy(desc(
            concat_ws("", col("like_count"), col("sub_comment_count"))
        ))
        
        # 获取三种类型的前50条评论
        likes_top = df.withColumn("rank", row_number().over(window_likes)) \
            .filter(col("rank") <= 50) \
            .select("car_id", "comment", "like_count", lit("0").alias("interaction_type"))
            
        comments_top = df.withColumn("rank", row_number().over(window_comments)) \
            .filter(col("rank") <= 50) \
            .select("car_id", "comment", "sub_comment_count", lit("1").alias("interaction_type"))
            
        total_top = df.withColumn("rank", row_number().over(window_total)) \
            .filter(col("rank") <= 50) \
            .select(
                "car_id", 
                "comment",
                concat_ws("", col("like_count"), col("sub_comment_count")).alias("interaction_count"),
                lit("2").alias("interaction_type")
            )
        
        # 合并结果
        result = likes_top.union(comments_top).union(total_top)
        
        return result
        
    except Exception as e:
        logger.error(f"处理表 {table_name} 时发生错误: {str(e)}")
        raise

def aggregate_car_comments():
    try:
        logger = setup_logger()
        start_time = time.time()
        logger.info("开始评论数据聚合处理")
        
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
        
        # 处理所有表并合并结果
        all_results = None
        processed_tables = 0
        failed_tables = 0
        
        for table_name in car_tables:
            table_start_time = time.time()
            try:
                result = process_table_comments(spark, table_name, mysql_properties, logger)
                
                if all_results is None:
                    all_results = result
                else:
                    all_results = all_results.union(result)
                
                table_time = time.time() - table_start_time
                logger.info(f"成功处理表 {table_name}，耗时 {table_time:.2f} 秒")
                processed_tables += 1
                
            except Exception as e:
                logger.error(f"处理表 {table_name} 时发生错误: {str(e)}")
                failed_tables += 1
                continue
        
        if all_results:
            try:
                # 将结果写入目标表
                all_results.write \
                    .jdbc(
                        url=mysql_properties["url"],
                        table="aa_spark_car_comments",
                        mode="overwrite",
                        properties=mysql_properties
                    )
                logger.info("成功将评论数据写入目标表")
            except Exception as e:
                logger.error(f"写入结果到目标表时发生错误: {str(e)}")
        else:
            logger.warning("没有成功处理任何数据表")
        
        # 记录总体统计信息
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"评论数据聚合处理完成")
        logger.info(f"总耗时: {total_time:.2f} 秒")
        logger.info(f"成功处理表数量: {processed_tables}")
        logger.info(f"失败表数量: {failed_tables}")
        
        spark.stop()
    except Exception as e:
        logger.error(f"执行聚合操作时出错: {str(e)}")
        raise

if __name__ == "__main__":
    aggregate_car_comments() 