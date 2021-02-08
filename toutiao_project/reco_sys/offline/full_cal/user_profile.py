#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/1/28 20:48                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 处理全量的用户行为日志
# * 1、创建HIVE基本数据表
# * 2、读取固定时间内的用户行为日志
# * 3、进行用户日志数据处理
# * 4、存储到user_article_basic表中
import os
import sys

BASE_DIR = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(BASE_DIR))

from offline import SparkSessionBase
class UpdateUserProfile(SparkSessionBase):
    SPARK_APP_NAME='updateUser'
    ENABLE_HIVE_SUPPORT=True

    SPARK_EXECUTEOR_MEMORY='2g'

    def __init__(self):
        self.spark = self._create_spark_session()

uup = UpdateUserProfile()

# 读取固定时间内的用户行为日志
uup.spark.sql('use profile')
user_action = uup.spark.sql("'select actionTime,readTime,channelId,param,articleId,param.algorithmCombine,param.action,param.userId from user_action where dt>='2019-04-04'")
user_action.show()

# +-------------------+--------+---------+--------------------+----------------+--------+-------------------+
# |         actionTime|readTime|channelId|           articleId|algorithmCombine|  action|             userId|
# +-------------------+--------+---------+--------------------+----------------+--------+-------------------+
# |2019-04-09 07:13:25|        |        0|[15716, 19171, 13...|              C2|exposure|1103195673450250240|
# |2019-04-09 07:13:49|        |       18|               19171|              C2|   click|1103195673450250240|
# |2019-04-09 07:14:02|   12238|       18|               19171|              C2|    read|1103195673450250240|
# |2019-04-09 07:14:04|        |       18|               13797|              C2|   click|1103195673450250240|

# 用户日志处理
def _compute(row):
    _list=[]
    if row.action=='exposure':
        for article_id in eval(row.articleId):
            # 用户ID和文章ID拼接一个样本
            # ['user_id','action_time','article_id','channel_id','shared','clicked','collected','exposure','read_time']
            _list.append([row.userId,row.actionTime,article_id,row.channelId,False,False,False,True,row.readTime])
        return _list

    else:
        class Temp():
            shared=False
            clicked=False
            collected=False
            read_time=''

        _tp = Temp()
        if row.action=='click':
            _tp.clicked=True
        elif row.action=='share':
            _tp.shared=True
        elif row.action=='collect':
            _tp.collected=True
        elif row.action=='read':
            _tp.clicked=True
        else:
            pass

        _list.append([row.userId,row.actionTime,int(row.articleId),row.channelId,_tp.shared,_tp.clicked,_tp.collected,True,row.readTime])
        return  _list

_res = user_action.rdd.flatMap(_compute)
user_action_basic = _res.toDF(['user_id','action_time','article_id','channel_id','shared','clicked','collected','exposure','read_time'])

user_action_basic.show()
# +-------------------+-------------------+----------+----------+------+-------+---------+--------+---------+
# |            user_id|        action_time|article_id|channel_id|shared|clicked|collected|exposure|read_time|
# +-------------------+-------------------+----------+----------+------+-------+---------+--------+---------+
# |1103195673450250240|2019-04-09 07:13:25|     15716|         0| false|  false|    false|    true|         |
# |1103195673450250240|2019-04-09 07:13:25|     19171|         0| false|  false|    false|    true|         |
# |1103195673450250240|2019-04-09 07:13:25|     13797|         0| false|  false|    false|    true|         |
# |1103195673450250240|2019-04-09 07:13:25|     17511|         0| false|  false|    false|    true|         |

# 存储到user_article_basic表中
old = uup.spark.sql('select * from user_article_basic')
new = old.unionAll(user_action_basic)# 增量更新才需要合并历史数据，全量更新其实不需要
# new.registerTempTable('temptable')

# uup.spark.sql("insert overwrite table user_article_basic select user_id, max(action_time) as action_time, "
#         "article_id, max(channel_id) as channel_id, max(shared) as shared, max(clicked) as clicked, "
#         "max(collected) as collected, max(exposure) as exposure, max(read_time) as read_time from temptable "
#         "group by user_id, article_id")

# 用户画像的关键词获取以及权重计算
# 读取user_article_basic表，合并行为表和文章画像中的主题词
uup.spark.sql('use profile')
user_basic = uup.spark.sql('select * from user_article_basic').drop('channel_id')
user_basic.show()
# +-------------------+-------------------+-------------------+------+-------+---------+--------+---------+
# |            user_id|        action_time|         article_id|shared|clicked|collected|exposure|read_time|
# +-------------------+-------------------+-------------------+------+-------+---------+--------+---------+
# |1105045287866466304|2019-03-11 18:13:45|              14225| false|  false|    false|    true|         |
# |1106476833370537984|2019-03-15 16:46:50|              14208| false|  false|    false|    true|         |
# |1109980466942836736|2019-03-25 08:50:36|              19233| false|  false|    false|    true|         |
# 读取文章画像
uup.spark.sql('use article')
article_topic = uup.spark.sql('select article_id,channel_id,topics from article_profile')
article_topic.show()
# +----------+----------+--------------------+
# |article_id|channel_id|              topics|
# +----------+----------+--------------------+
# |        26|        17|[Electron, 全自动, 产...|
# |        29|        17|[WebAssembly, 影音,...|
# |       474|        17|[textAlign, borde...|
# |       964|        11|[protocol, RMI, d...|
user_topic = user_basic.join(article_topic,on=['article_id'],how='left')
user_topic.show()
# +----------+-------------------+-------------------+------+-------+---------+--------+---------+----------+--------------------+
# |article_id|            user_id|        action_time|shared|clicked|collected|exposure|read_time|channel_id|              topics|
# +----------+-------------------+-------------------+------+-------+---------+--------+---------+----------+--------------------+
# |     13401|                 10|2019-03-06 10:06:12| false|  false|    false|    true|         |        18|[补码, 字符串, 李白, typ...|
# |     13401|1114864237131333632|2019-04-09 16:39:51| false|  false|    false|    true|         |        18|[补码, 字符串, 李白, typ...|
# |     13401|1106396183141548032|2019-03-28 10:58:20| false|  false|    false|    true|         |        18|[补码, 字符串, 李白, typ...|

import pyspark.sql.functions as F
user_topic = user_topic.withColumn('topic',F.explode('topics')).drop('topics')
user_topic.show()
# +----------+-------------------+-------------------+------+-------+---------+--------+---------+----------+--------+
# |article_id|            user_id|        action_time|shared|clicked|collected|exposure|read_time|channel_id|   topic|
# +----------+-------------------+-------------------+------+-------+---------+--------+---------+----------+--------+
# |     13401|                 10|2019-03-06 10:06:12| false|  false|    false|    true|         |        18|      补码|
# |     13401|                 10|2019-03-06 10:06:12| false|  false|    false|    true|         |        18|     字符串|

def compute_user_label_weights(partitons):
    '''
    计算用户关键词权重
    :param partitons:
    :return:
    '''
    weightsOfaction={
        'read_min':1,
        'read_middle':2,
        'collect':2,
        'share':3,
        'click':5
    }

    #导入包
    from datetime import datetime
    import numpy as np

    # 循环每个用户对每个关键词的处理
    for row in partitons:
        # 计算时间系数
        t = datetime.now()-datetime.strptime(row.action_time,'%Y-%m-%d %H:%M:%S')
        alpha = 1/(np.log(t.days+1)+1)

        # 判断一下这个关键词对应的操作文章时间大小的权重处理
        if row.read_time=='':
            read_t = 0
        else:
            read_t=int(row.read_time)

        # 阅读时间的行为分数计算出来
        read_score = weightsOfaction['read_middle'] if read_t>1000 else weightsOfaction['read_min']

        # 计算row.topic的权重
        weights = alpha*(row.shared*weightsOfaction['share']+row.clicked*weightsOfaction['click']+
                         row.collected*weightsOfaction['collect']+read_score)
        # user_profile base 表
        import happybase
        import json

        pool = happybase.ConnectionPool(size=3, host='10.0.80.13')
        with pool.connection() as conn:
            table = conn.table('user_profile')
            table.put('user:{}'.format(row.user_id).encode(),{'partial:{}:{}'.format(row.channel_id,row.topic).encode():json.dumps(weights).encode()})

user_topic.foreachPartiton(compute_user_label_weights)

