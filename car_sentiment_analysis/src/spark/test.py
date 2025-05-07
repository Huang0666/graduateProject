from pyspark.sql import SparkSession

# 创建Spark会话
spark = SparkSession.builder \
    .appName("ExampleApp") \
    .getOrCreate()

# 创建示例数据
data = [("a", 1), ("b", 2), ("c", 3)]
columns = ["letter", "number"]

# 创建DataFrame
df = spark.createDataFrame(data, columns)

# 显示DataFrame内容
df.show()

# 停止Spark会话
spark.stop()
