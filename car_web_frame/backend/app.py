from flask import Flask, jsonify, request
from flask_cors import CORS
import pymysql
from dotenv import load_dotenv
import os

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用CORS

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'db': 'media_crawler_raw_data_dy',
    'charset': 'utf8mb4'
}

# 数据库连接函数
def get_db_connection():
    return pymysql.connect(**DB_CONFIG)

# 测试路由
@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'message': 'Backend is working!'})

# 获取所有车型
@app.route('/api/cars', methods=['GET'])
def get_cars():
    try:
        conn = get_db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute('SELECT car_id, car_name FROM aa_spark_car_info')
            cars = cursor.fetchall()
            print(f"获取到的车型数据: {cars}")  # 添加日志
        conn.close()
        return jsonify(cars)
    except Exception as e:
        print(f"获取车型数据时出错: {str(e)}")  # 添加错误日志
        return jsonify({'error': str(e)}), 500

# 获取特定车型的评价统计
@app.route('/api/car/<int:car_id>/evaluation-stats', methods=['GET'])
def get_car_evaluation_stats(car_id):
    try:
        conn = get_db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # 先检查车型是否存在
            cursor.execute('SELECT car_name FROM aa_spark_car_info WHERE car_id = %s', (car_id,))
            car = cursor.fetchone()
            print(f"查找的车型ID: {car_id}, 找到的车型: {car}")  # 添加日志

            if not car:
                print(f"未找到车型ID: {car_id}")  # 添加日志
                return jsonify({'error': 'Car not found'}), 404

            # 获取评价统计
            cursor.execute('''
                SELECT 
                    positive_evaluation_count,
                    negative_evaluation_count,
                    neutral_evaluation_count,
                    irrelevant_evaluation_count
                FROM aa_spark_car_evaluation_stats 
                WHERE car_id = %s
            ''', (car_id,))
            stats = cursor.fetchone()
            print(f"获取到的评价统计数据: {stats}")  # 添加日志

            if not stats:
                print(f"未找到车型ID {car_id} 的评价统计数据")  # 添加日志
                return jsonify({'error': 'No evaluation stats found'}), 404

        conn.close()
        return jsonify(stats)
    except Exception as e:
        print(f"获取评价统计数据时出错: {str(e)}")  # 添加错误日志
        return jsonify({'error': str(e)}), 500

# 获取特定车型的评论数据
@app.route('/api/car/<int:car_id>/comments/<string:interaction_type>', methods=['GET'])
def get_car_comments(car_id, interaction_type):
    try:
        conn = get_db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute('''
                SELECT * FROM aa_spark_car_comments 
                WHERE car_id = %s AND interaction_type = %s
                ORDER BY interaction_count DESC
                LIMIT 50
            ''', (car_id, interaction_type))
            comments = cursor.fetchall()
            print(f"获取到的评论数据数量: {len(comments)}")  # 添加日志
        conn.close()
        return jsonify(comments)
    except Exception as e:
        print(f"获取评论数据时出错: {str(e)}")  # 添加错误日志
        return jsonify({'error': str(e)}), 500

# 获取车型类别评论数据
@app.route('/api/car-type/comments', methods=['GET'])
def get_car_type_comments():
    try:
        category = request.args.get('category')
        interaction_type = request.args.get('interaction_type')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 10))
        
        if not category or interaction_type is None:
            return jsonify({'error': 'Missing required parameters'}), 400

        conn = get_db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # 获取总数
            cursor.execute('''
                SELECT COUNT(*) as total
                FROM aa_spark_car_type_comments
                WHERE category_description = %s AND interaction_type = %s
            ''', (category, interaction_type))
            total = cursor.fetchone()['total']

            # 获取分页数据
            offset = (page - 1) * page_size
            cursor.execute('''
                SELECT content, interaction_count
                FROM aa_spark_car_type_comments
                WHERE category_description = %s AND interaction_type = %s
                ORDER BY interaction_count DESC
                LIMIT %s OFFSET %s
            ''', (category, interaction_type, page_size, offset))
            comments = cursor.fetchall()

        conn.close()
        return jsonify({
            'total': total,
            'page': page,
            'page_size': page_size,
            'data': comments
        })
    except Exception as e:
        print(f"获取车型类别评论数据时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 获取车型类别的评价统计数据
@app.route('/api/car-type/evaluation-stats', methods=['GET'])
def get_car_type_evaluation_stats():
    try:
        category = request.args.get('category')
        if not category:
            return jsonify({'error': 'Category is required'}), 400

        conn = get_db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # 首先获取该类型的所有车辆ID
            cursor.execute('''
                SELECT car_id 
                FROM aa_spark_car_info 
                WHERE car_type = %s OR power_type = %s
            ''', (category, category))
            car_ids = cursor.fetchall()
            
            if not car_ids:
                return jsonify({'error': 'No cars found for this category'}), 404

            # 获取这些车辆的评价统计并求和
            car_id_list = [car['car_id'] for car in car_ids]
            placeholders = ', '.join(['%s'] * len(car_id_list))
            
            cursor.execute(f'''
                SELECT 
                    SUM(positive_evaluation_count) as total_positive,
                    SUM(negative_evaluation_count) as total_negative,
                    SUM(neutral_evaluation_count) as total_neutral,
                    SUM(irrelevant_evaluation_count) as total_irrelevant
                FROM aa_spark_car_evaluation_stats
                WHERE car_id IN ({placeholders})
            ''', tuple(car_id_list))
            
            stats = cursor.fetchone()
            
            if not stats or not any(stats.values()):
                return jsonify({'error': 'No evaluation data found'}), 404

        conn.close()
        return jsonify(stats)
    except Exception as e:
        print(f"获取车型类别评价统计时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 获取所有车型名称
@app.route('/api/cars/names', methods=['GET'])
def get_car_names():
    try:
        conn = get_db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute('SELECT car_id, car_name FROM aa_spark_car_info')
            cars = cursor.fetchall()
        conn.close()
        return jsonify(cars)
    except Exception as e:
        print(f"获取车型名称时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 获取单个车型的关键词
@app.route('/api/car/keywords', methods=['GET'])
def get_car_keywords():
    try:
        car_id = request.args.get('car_id')
        sentiment_type = request.args.get('sentiment_type')
        
        if not car_id or sentiment_type is None:
            return jsonify({'error': 'Missing required parameters'}), 400

        conn = get_db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute('''
                SELECT keywords
                FROM aa_spark_car_keywords
                WHERE car_id = %s AND sentiment_type = %s
            ''', (car_id, sentiment_type))
            keywords = cursor.fetchall()
        conn.close()
        
        return jsonify(keywords)
    except Exception as e:
        print(f"获取车型关键词时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 获取车型类别的关键词
@app.route('/api/car-type/keywords', methods=['GET'])
def get_car_type_keywords():
    try:
        category = request.args.get('category')
        sentiment_type = request.args.get('sentiment_type')
        
        if not category or sentiment_type is None:
            return jsonify({'error': 'Missing required parameters'}), 400

        conn = get_db_connection()
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute('''
                SELECT keyword
                FROM aa_spark_car_type_keywords
                WHERE category_description = %s AND sentiment_type = %s
            ''', (category, sentiment_type))
            keywords = cursor.fetchall()
        conn.close()
        
        return jsonify(keywords)
    except Exception as e:
        print(f"获取车型类别关键词时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 