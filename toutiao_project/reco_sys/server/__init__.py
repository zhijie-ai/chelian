#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/7 22:24                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import happybase
from setting.default import DefaultConfig
import redis

pool = happybase.ConnectionPool(size=10, host='hadoop-master', port=9090)

# 召回数据
redis_client = redis.StrictRedis(host=DefaultConfig.REDIS_HOST,
                                 port=DefaultConfig.REDIS_PORT,
                                 db=10,
                                 decode_responses=True)
# 用于缓存的Redis数据库
cache_client = redis.StrictRedis(host=DefaultConfig.REDIS_HOST,
                                 port=DefaultConfig.REDIS_PORT,
                                 db=8,
                                 decode_responses=True)

# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# # spark配置
# conf = SparkConf()
# conf.setAll(DefaultConfig.SPARK_GRPC_CONFIG)
#
SORT_SPARK = SparkSession.builder.config(conf=conf).getOrCreate()