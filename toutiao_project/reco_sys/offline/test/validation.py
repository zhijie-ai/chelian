#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/1/11 16:19                       #
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

from pyspark.ml.feature import CountVectorizerModel
from pyspark.ml.feature import IDFModel

ktt = KeywordsToTfidf()
cv_model = CountVectorizerModel.load("hdfs:///recommendation/models/CV.model")
idf_model = IDFModel.load("hdfs:///recommendation/models/IDF.model")
keywords_list_with_idf = list(zip(cv_model.vocabulary, idf_model.idf.toArray()))
print(keywords_list_with_idf[0:10])
# [('&#', 1.4179305192489027), ('代码', 0.682506493926397), ('方法', 0.7083862624653089), ('数据', 0.834664504878301), ('对象', 1.2257424177824428), ('函数', 1.3010506422277146), ('return', 1.1586225696610128), ('文件', 1.1119647845142948), ('name', 1.3839902789969114), ('this', 1.6221988413776336)]
def func(data):
    for index in range(len(data)):
        data[index] = list(data[index])
        data[index].append(index)
        data[index][1] = float(data[index][1])
func(keywords_list_with_idf)
print(keywords_list_with_idf[0:5])