#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/4 21:17                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from setting.default import DefaultConfig

# 1 创建spark streaming context
conf = SparkConf()
conf.setAll(DefaultConfig.SPARK_ONLINE_CONFIG)
sc = SparkContext(conf=conf)
stream_sc = StreamingContext(sc,60)

# 2.配置与kafka读取的配置
similar_kafka = {'metadata.broker.list':DefaultConfig.KAFKA_SERVER,'group.id':'similar'}
SIMILAR_DS = KafkaUtils.createDirectStream(stream_sc,['click-trace'],similar_kafka)

# 配置hot文章读取配置
kafka_params = {'metadata.broker.list':DefaultConfig.KAFKA_SERVER}# 用的是默认的组
HOT_DS = KafkaUtils.createDirectStream(stream_sc,['click-trace'],kafka_params)

#配置新文章的读取
NEW_ARTICLE_DS = KafkaUtils.createDirectStream(stream_sc,['new-article'],kafka_params)