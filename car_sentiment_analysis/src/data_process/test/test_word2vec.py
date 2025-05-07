"""
测试词向量模型加载和使用
"""
import logging
import os
from gensim.models import KeyedVectors
from pathlib import Path

def test_word2vec():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 模型路径
    model_path = Path(__file__).parent.parent.parent / 'data/pretrained/word2vec/tencent_small/1000000-small.txt'
    
    try:
        # 检查文件是否存在
        if not model_path.exists():
            raise FileNotFoundError(f"词向量文件不存在：{model_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # 转换为MB
        logging.info(f"词向量文件大小：{file_size:.2f}MB")
        
        # 检查文件前几行
        logging.info("检查文件格式：")
        with open(model_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            logging.info(f"文件第一行：{first_line}")
            second_line = f.readline().strip()
            logging.info(f"文件第二行：{second_line}")
        
        # 加载模型
        logging.info(f"开始加载词向量模型：{model_path}")
        model = KeyedVectors.load_word2vec_format(
            str(model_path), 
            binary=False,
            unicode_errors='ignore'  # 添加错误处理
        )
        logging.info("模型加载成功！")
        
        # 输出模型基本信息
        logging.info(f"词表大小：{len(model.key_to_index)}")
        logging.info(f"向量维度：{model.vector_size}")
        
        # 测试词向量操作
        test_words = ['汽车', '发动机', '轮胎', '方向盘']
        
        # 1. 测试词向量获取
        logging.info("\n测试词向量获取:")
        for word in test_words:
            if word in model:
                vector = model[word]
                logging.info(f"词语 '{word}' 的向量维度: {vector.shape}")
            else:
                logging.warning(f"词语 '{word}' 不在词表中")
        
        # 2. 测试相似词查找
        logging.info("\n测试相似词查找:")
        for word in test_words:
            if word in model:
                similar_words = model.most_similar(word, topn=5)
                logging.info(f"\n'{word}'的前5个最相似词:")
                for similar_word, score in similar_words:
                    logging.info(f"  {similar_word}: {score:.4f}")
            else:
                logging.warning(f"词语 '{word}' 不在词表中")
        
        # 3. 测试词语相似度
        logging.info("\n测试词语相似度:")
        word_pairs = [
            ('汽车', '轿车'),
            ('发动机', '引擎'),
            ('方向盘', '车轮')
        ]
        for word1, word2 in word_pairs:
            if word1 in model and word2 in model:
                similarity = model.similarity(word1, word2)
                logging.info(f"'{word1}' 和 '{word2}' 的相似度: {similarity:.4f}")
            else:
                logging.warning(f"词对 ({word1}, {word2}) 中存在词不在词表中")
        
        logging.info("\n所有测试完成！")
        
    except Exception as e:
        logging.error(f"测试过程中出现错误：{str(e)}")
        # 打印详细的错误堆栈
        import traceback
        logging.error(f"错误详情：\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    test_word2vec() 