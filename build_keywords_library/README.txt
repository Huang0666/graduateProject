#构建停用词
|-build_keyword_library
    |- data_storage 存储csv和txt
    |- data_conduct
        |-mysql_data_reverse_csv : mysql中content列转出到CSV--> .txt
        |-find_keywords : 用于统计词频，从高频次选出车评关键词
        |-tyc.txt :find_keywords 停用词
        |-keywords：存储关键词，用于最后的正则筛选
        |-merge.txt  临时存储关键字

#统计完高频词将其分类，保存成可匹配的格式，用于mysql筛选数据
#筛选完数据，开始情感分析