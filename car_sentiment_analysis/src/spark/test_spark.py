from pyspark.sql import SparkSession
import os
import sys

def test_spark_setup():
    try:
        print("开始测试Spark环境配置...")
        
        # 获取当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"当前目录: {current_dir}")
        
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
        print("正在创建SparkSession...")
        spark = SparkSession.builder \
            .appName("SparkTest") \
            .master("local[*]") \
            .getOrCreate()
            
        print("SparkSession创建成功!")
        
        # 创建测试数据
        test_data = [("测试1", 1), ("测试2", 2), ("测试3", 3)]
        df = spark.createDataFrame(test_data, ["名称", "值"])
        
        print("\n测试数据预览:")
        df.show()
        
        print("\nSparkSession配置信息:")
        print(f"Spark版本: {spark.version}")
        print(f"应用ID: {spark.sparkContext.applicationId}")
        print(f"主节点: {spark.sparkContext.master}")
        
        spark.stop()
        print("\n测试完成！Spark环境配置正确。")
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    test_spark_setup() 