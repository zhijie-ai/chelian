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

ktt = KeywordsToTfidf()

ktt.spark.sql('use article')
profile = ktt.spark.sql("select * from article_profile limit 10")
profile.registerTempTable("incremental")
df = ktt.spark.sql('select article_id,channel_id,keyword,weight from incremental LATERAL VIEW explode(keyword) AS keyword,weight')
df.show(truncate=False)

