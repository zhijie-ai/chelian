#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/1/10 17:02                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import os
import sys
sys.path.insert(0,'/usr/local/service/spark/python/lib/pyspark.zip')
sys.path.insert(0,'/usr/local/service/spark/python/lib/py4j-0.10.7-src.zip')
sys.path.insert(0,'/usr/local/service/spark/jars/spark-core_2.11-2.3.2.jar')

# 如果当前代码文件运行测试需要加入修改路径，避免出现后导包问题
BASE_DIR = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(BASE_DIR))
from offline import SparkSessionBase




class OriginArticleData(SparkSessionBase):


    SPARK_APP_NAME = "mergeArticle"
    SPARK_URL = "yarn"

    ENABLE_HIVE_SUPPORT = True

    def __init__(self):
        self.spark = self._create_spark_session()

oa = OriginArticleData()

# 进行文章 前两个表 的合并
oa.spark.sql("use toutiao")
# news_article_basic 与news_article_content, article_id
basic_content = oa.spark.sql("select a.article_id, a.channel_id, a.title, b.content from news_article_basic a inner join news_article_content b on a.article_id=b.article_id")

import pyspark.sql.functions as F
import gc

# 增加channel的名字，后面会使用
basic_content.registerTempTable("temparticle")
channel_basic_content = oa.spark.sql(
  "select t.*, n.channel_name from temparticle t left join news_channel n on t.channel_id=n.channel_id")

# 利用concat_ws方法，将多列数据合并为一个长文本内容（频道，标题以及内容合并）
oa.spark.sql('use article')
sentence_df = channel_basic_content.select('article_id','channel_id','channel_name','title','content',
                                           F.concat_ws(',',
                                                       channel_basic_content.channel_name,
                                                       channel_basic_content.title,
                                                       channel_basic_content.content).alias('sentence'))

del basic_content
del channel_basic_content
gc.collect()

sentence_df.write.insertInto("article_data")

# Tfidf计算


