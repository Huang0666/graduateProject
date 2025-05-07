#该文件用于词频统计
import jieba
from collections import Counter

# 文件路径配置
input_file = "audi_a6_export_20250430_150715.csv"  # 待分析的中文文本路径
stopwords_file = "tyc.txt"  # 停用词表路径（需自行准备）

# 读取文本内容
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# 加载停用词表
with open(stopwords_file, "r", encoding="utf-8") as f:
    stopwords = set([line.strip() for line in f])

# 中文分词处理
words = jieba.lcut(text)

# 词频统计（过滤单字和停用词）
word_counts = Counter()
for word in words:
    if len(word) > 1 and word not in stopwords:
        word_counts[word] += 1

# 获取前20个高频词
top_words = word_counts.most_common(500)

# 结果输出
print("{:<10}{:>8}".format("词语", "频次"))
print("-------------------")
for word, count in top_words:
    print(f"{word:<10}{count:>8}")
    with open("merge.txt", "w", encoding="utf-8") as file:
        for word, count in top_words:
            file.write(f"{word:<10}\n")
