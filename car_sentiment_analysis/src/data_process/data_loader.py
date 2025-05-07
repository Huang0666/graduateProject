"""
数据加载模块：从MySQL数据库加载评论数据并保存为CSV文件
第一次：加载训练数据-car_id(1-10)的10款车型各100条数据到CSV
第二次：加载训练数据-car_id(1-10)的10款车型各1000条数据到CSV

"""

import csv
import os
import pymysql
from pymysql import MySQLError

# 数据库配置
db_config = {
    'host': 'localhost',
    'database': 'media_crawler_raw_data_dy',
    'user': 'root',
    'password': 'root'
}

# 数据表名列表
tables = [
    "toyota_corolla",
    "nissan_sylphy",
    "volvo_s60",
    "volkswagen_lavida",
    "buick_excelle_gt",
    "volkswagen_sagitar",
    "volkswagen_passat",
    "cadillac_ct4",
    "chery_arrizo_8",
    "honda_civic"
]

# 表名 -> 中文车名映射
car_name_map = {
    "toyota_corolla": "丰田卡罗拉",
    "nissan_sylphy": "日产轩逸",
    "volvo_s60": "沃尔沃S60",
    "volkswagen_lavida": "大众朗逸",
    "buick_excelle_gt": "别克英朗",
    "volkswagen_sagitar": "大众速腾",
    "volkswagen_passat": "大众帕萨特",
    "cadillac_ct4": "凯迪拉克CT4",
    "chery_arrizo_8": "艾瑞泽8",
    "honda_civic": "思域"
}

# 输出文件 - 使用绝对路径
base_dir = "D:/graduation_project/car_sentiment_analysis/data/raw"
# 输出文件名   修改这里可以修改输出的文件名
csv_filename = os.path.join(base_dir, "all_raw_cars_comments_supplement_10000.csv")

# 确保输出目录存在
try:
    os.makedirs(base_dir, exist_ok=True)
    # 测试文件写入权限
    with open(csv_filename, 'w') as test_file:
        pass
except PermissionError:
    print(f"❌ 没有写入权限，请检查文件夹权限: {base_dir}")
    exit(1)
except Exception as e:
    print(f"❌ 创建目录或测试文件时出错: {e}")
    exit(1)

try:
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='|')
        headers = ['id', 'car_name', 'content', 'like_count', 'sub_comment_count', 'sentiment_analysis_results']
        csv_writer.writerow(headers)

        global_id = 1

        for table in tables:
            car_name = car_name_map.get(table, "未知车型")

            # 修改这里可以修改每次查询的条数    
            query = f"SELECT content, like_count, sub_comment_count FROM {table} WHERE content IS NOT NULL AND content != '' ORDER BY RAND() LIMIT 1000"
            cursor.execute(query)
            records = cursor.fetchall()

            for row in records:
                content = row[0].strip()
                like_count = row[1] or 0
                sub_comment_count = row[2] or 0

                # 拼接车型前缀
                content_with_car = f"{car_name}：{content}"
                sentiment_result = ""

                csv_writer.writerow([
                    global_id, car_name, content_with_car, like_count, sub_comment_count, sentiment_result
                ])
                global_id += 1

    print(f"✅ 成功写入 {csv_filename}，共计 {global_id - 1} 条数据")

except MySQLError as e:
    print(f"❌ 数据库连接或查询出错: {e}")
except FileNotFoundError:
    print(f"❌ 无法创建或访问文件路径: {csv_filename}")
except Exception as e:
    print(f"❌ 发生未知错误: {e}")

finally:
    try:
        if 'connection' in locals() and connection.open:
            cursor.close()
            connection.close()
            print("🔌 数据库连接已关闭")
    except Exception as e:
        print(f"❌ 关闭数据库连接时出错: {e}")