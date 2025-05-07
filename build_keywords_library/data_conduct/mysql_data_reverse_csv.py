#该脚本用于将数据库MySQL文件提取到CSV
import pymysql
import csv
from datetime import datetime

# 1.
# 连接MySQL数据库


conn = pymysql.connect(host='localhost',
                       user='root',
                       password='root',
                       database='media_crawler_raw_data_dy',
                       charset='utf8mb4',
                       cursorclass=pymysql.cursors.DictCursor)

# 2.
# 执行SQL查询并导出CSV
try:
    with conn.cursor() as cursor:
        sql = "SELECT content FROM buick_envision"
        cursor.execute(sql)
        results = cursor.fetchall()

# 生成带时间戳的CSV文件名（避免覆盖）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"audi_a6_export_{timestamp}.csv"

        # 写入CSV文件
        with open(csv_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
            if results:
                # 获取字段名作为CSV表头
                fieldnames = results[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                # 逐行写入数据
                for row in results:
                    writer.writerow(row)
                print(f"数据已导出到: {csv_filename}")
            else:
                print("警告：未查询到任何数据！")
finally: conn.close()  # 确保关闭数据库连接