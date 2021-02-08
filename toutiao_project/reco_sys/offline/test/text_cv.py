#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/1/11 14:09                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import os
import sys
import gc

sys.path.insert(0,'/usr/local/service/spark/python/lib/pyspark.zip')
sys.path.insert(0,'/usr/local/service/spark/python/lib/py4j-0.10.7-src.zip')
sys.path.insert(0,'/usr/local/service/spark/jars/spark-core_2.11-2.3.2.jar')
# 如果当前代码文件运行测试需要加入修改路径，避免出现后导包问题
BASE_DIR = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(BASE_DIR))
from offline import SparkSessionBase

class KeywordsToTfidf(SparkSessionBase):
    SPARK_APP_NAME = "keywordsByTFIDF"
    #SPARK_EXECUTOR_MEMORY = "7g"

    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = self._create_spark_session()

def segmentation(partition):
    import os
    import re

    import jieba
    import jieba.analyse
    import jieba.posseg as pseg
    import codecs

    # abspath = '/recommendation/words/ITKeywords.txt'
    # jieba.load_userdict(abspath)

    # 停用词
    # stopwords_path = os.path.join(abspath, "stopwords.txt")
    # def get_stopwords_list():
    #     stopwords_list = [i.strip() for i in codecs.open(stopwords_path).readlines()]
    #     return stopwords_list

    def cut_sentence(sentence):
        '''对切割之后的词语进行过滤，去除停用词，保留名词，英文和自定义词库中的词，长度大于2的词'''
        seg_list = pseg.lcut(sentence)
        # seg_list = [i for i in seg_list if i.flag not in stopwords_list]#i.word,i.flag
        filtered_words_list=[]
        for seg in seg_list:
            if len(seg.word) <=1:
                continue
            elif seg.flag=='eng':
                if len(seg.word)<=2:
                    continue
                else:
                    filtered_words_list.append(seg.word)
            elif seg.flag.startswith("n"):
                filtered_words_list.append(seg.word)
            elif seg.flag in ["x", "eng"]:  # 是自定一个词语或者是英文单词
                filtered_words_list.append(seg.word)
        return filtered_words_list

    for row in partition:
        sentence = re.sub("<.*?>", "", row.sentence)
        words  = cut_sentence(sentence)
        yield row.article_id,row.channel_id,words


ktt = KeywordsToTfidf()

ktt.spark.sql('use article')
article_dataframe = ktt.spark.sql("select * from article_data limit 1000")
words_df = article_dataframe.rdd.mapPartitions(segmentation).toDF(['article_id','channel_id','words'])

# 词语与词频统计
from pyspark.ml.feature import CountVectorizer
cv = CountVectorizer(inputCol='words',outputCol='countFeatures',vocabSize=200*10000,minDF=1.0)
# 训练词频统计模型
cv_model = cv.fit(words_df)
# 得出词频向量结果
cv_result = cv_model.transform(words_df)
cv_result.printSchema()
# root
#  |-- article_id: long (nullable = true)
#  |-- channel_id: long (nullable = true)
#  |-- words: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- countFeatures: vector (nullable = true)
print(cv_model.vocabulary[0:10])
print(dir(cv_model))
# idf_model.idf.toArray()
from pyspark.ml.feature import IDF
idf = IDF(inputCol='countFeatures',outputCol='idfFeatures')
idf_model = idf.fit(cv_result)
tfidf_result = idf_model.transform(cv_result)
print('====================')
# print(idf_model.vocabulary[0:10])# 出错，没有vocabulary
print(dir(idf_model))
tfidf_result.printSchema()
# root
#  |-- article_id: long (nullable = true)
#  |-- channel_id: long (nullable = true)
#  |-- words: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- countFeatures: vector (nullable = true)
#  |-- idfFeatures: vector (nullable = true)
# row.idfFeatures.indices, row.idfFeatures.values
# vector中有indices和values属性