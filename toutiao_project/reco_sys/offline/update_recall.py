#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/1/31 20:03                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 离线用户召回
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))# toutiao_project
sys.path.insert(0, os.path.join(BASE_DIR))
from offline import SparkSessionBase
import time

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

from pyspark.ml.recommendation import ALS
from datetime import datetime
import numpy as np

class UpdateRecall(SparkSessionBase):
    SPARK_APP_NAME='updateRecall'
    ENABLE_HIVE_SUPPORT=True

    def __init__(self,number):
        self.spark = self._create_spark_session()
        self.N=number

    def update_als_recall(self):
        """
        更新基于模型(ALS)的协同过滤召回集
        :return:
        """
        # 读取用户行为基本表
        self.spark.sql('use profile')
        user_article_click = self.spark.sql('select * from user_article_basic').select(['user_id','article_id','clicked'])

        # 更换类型
        def change_type(row):
            return row.user_id,row.article_id,int(row.clicked)

        user_article_click = user_article_click.rdd.map(change_type).toDF(['user_id','article_id','clicked'])
        # 用户和文章ID超过ALS最大整数值,需要使用StringIndexer进行转换
        user_id_indexer = StringIndexer(inputCol='user_id',outputCol='als_user_id')
        article_id_indexer = StringIndexer(inputCol='article_id',outputCol='als_article_id')
        pip = Pipeline(stages=[user_id_indexer,article_id_indexer])
        pip_fit = pip.fit(user_article_click)
        als_user_article_click = pip_fit.transform(user_article_click)

        #模型训练和推荐默认每个用户固定文章个数
        als = ALS(userCol='als_user_id',itemCol='als_article_id',ratingCol='clicked',checkpointInterval=1)
        model = als.fit(als_user_article_click)
        recall_res = model.recommendationForAllUsers(self.N)

        # recall_res 得到需要使用StringInderer变换后的下标
        # 保存原来的下标映射关系
        refection_user = als_user_article_click.groupBy(['user_id']).max('als_user_id').withColumnRenamed('max(als_user_id)','als_user_id')
        refection_article = als_user_article_click.groupBy(['article_id']).max('als_article_id').withColumnRenamed('max(als_article_id','als_article_id')
        # Join推荐结果与 refection_user映射关系表
        # +-----------+--------------------+-------------------+
        # | als_user_id | recommendations | user_id |
        # +-----------+--------------------+-------------------+
        # | 8 | [[163, 0.91328144]... | 2 |
        #        | 0 | [[145, 0.653115], ... | 1106476833370537984 |
        recall_res = recall_res.join(refection_user,on=['als_user_id'],how='left').select(['als_user_id','recommendations','user_id'])
        # Join推荐结果与 refection_article映射关系表
        # +-----------+-------+----------------+
        # | als_user_id | user_id | als_article_id |
        # +-----------+-------+----------------+
        # | 8 | 2 | [163, 0.91328144] |
        # | 8 | 2 | [132, 0.91328144] |
        import pyspark.sql.functions as F
        recall_res = recall_res.withColumn('als_user_id',F.explode('recommendations')).drop('recommendations')
        # +-----------+-------+--------------+
        # | als_user_id | user_id | als_article_id |
        # +-----------+-------+--------------+
        # | 8 | 2 | 163 |
        # | 8 | 2 | 132 |
        def _article_id(row):
            return row.als_user_id,row.user_id,row.als_article_id[0]

        als_recall = recall_res.rdd.map(_article_id).toDF(['als_user_id','user_id','als_article_id'])
        als_recall = als_recall.join(refection_article,on=['als_article_id'],how='left').select(['user_id','article_id'])
        # 得到每个用户id对应的推荐文章
        # +-------------------+----------+
        # | user_id | article_id |
        # +-------------------+----------+
        # | 1106476833370537984 | 44075 |
        # | 1 | 44075 |
        # 每组统计每个用户推荐列表
        # als_recall = als_recall.groupBy('user_id').agg(F.collect_list('article_id')).withColumnRenamed('collect_list(article_id','article_list')
        self.spark.sql('use toutiao')
        news_article_basic = self.spark.sql('select article_id,channel_id from news_article_basic')
        als_recall = als_recall.join(news_article_basic,on='article_id',how='left')
        als_recall = als_recall.groupBy(['user_id','channel_id']).agg(F.collect_list('article_id')).withColumnRenamed('collect_list(article_id)','article_list')
        als_recall =als_recall.dropna()

        # 存储
        def save_offline_recall_hbase(partition):
            """
            离线模型召回结果存储
            :param partition:
            :return:
            """
            import happyhbase
            pool = happyhbase.ConnectionPool(size=10,host='192.168.19.137',port=9090)
            with pool.connection() as conn:
                for row in partition:
                    # 获取历史看过的该频道文章
                    history_recall = conn.table('history_recall')
                    # 多个版本
                    data = history_recall.cells('reco:his:{}'.format(row.user_id).encode(),
                                                'channel:{}'.format(row.channel_id).encode())
                    history=[]
                    if len(data)>2:
                        for l in data[:-1]:
                            history.extend(eval(l))
                    else:
                        history=[]


                    # 过滤reco_article与history
                    reco_res = list(set(row.article_list)-set(history))

                    if reco_res:
                        table = conn.table('cb_recall')
                        # 默认放在推荐频道
                        table.put('recall_user:{}'.format(row.user_id).encode(),
                                  {'als:{}'.format(row.channel_id).encode():str(reco_res).encode()})

                        # 放入历史推荐过文章
                        history_tabel = table.put('reco:his:{}'.format(row.user_id).encode(),
                                                  {'channel:{}'.format(row.channel_id):str(reco_res).encode()})

        als_recall.foreachPartition(save_offline_recall_hbase)


    def update_content_recall(self):
        """
        更新基于内容(画像)的推荐召回集.word2vec相似
        :return:
        """
        # 基于内容相似召回(画像召回)
        self.spark.sql('use profile')
        user_article_basic = self.spark.sql('select * from user_article_basic')
        user_article_basic = user_article_basic.filter('clicked=True')

        def save_content_filter_history_to_recall(partition):
            """
            计算每个用户的每个操作文章的相似文章，过滤之后，写入content召回表
            :param partition:
            :return:
            """
            import happybase
            pool = happybase.ConnectionPool(size=10,host='192.168.19.137',port=9090)
            with pool.connection() as conn:
                # key:article_id    ,column:similar:article_id
                similar_table=conn.table('article_similar')
                # 循环partition
                for row in partition:
                    # 获取相似文章结果表
                    similar_article = similar_table.row(str(row.article_id).encode(),
                                                        columns=[b'similar'])
                    # 相似文章相似度排序过滤,召回不需要太大的数据，百个，千
                    _srt = sorted(similar_article.items(),key=lambda item:item[1],reverse=True)
                    if _srt:
                        # 每次行为推荐十篇文章
                        reco_article = [int(i[0].split(b':')[1]) for i in _srt][:10]

                        # 获取历史看过的该频道文章
                        history_table = conn.table('history_recall')
                        # 多个版本
                        data = history_table.cells('reco:his:{}'.format(row.user_id).encode(),
                                                   'channel:{}'.format(row.channel_id).encode())
                        history = []
                        if len(data) >= 2:
                            for l in data[:-1]:
                                history.extend(eval(l))
                        else:
                            history = []

                        # 过滤reco_article与history
                        reco_res = list(set(reco_article)-set(history))

                        # 进行推荐，放入基于内容的召回表当中以及历史看过的文章表当中
                        if reco_res:
                            # content_table = conn.table('cb_content_recall')
                            content_table = conn.table('cb_recall')
                            content_table.put('recall:user:{}'.format(row.user_id).encode(),
                                              {'content:{}'.format(row.channel_id).encode():str(reco_res).encode()})

                            # 放入历史推荐过文章
                            history_table.put('reco:his:{}'.format(row.user_id).encode(),
                                              {'channel:{}'.format(row.channel_id).encode():str(reco_res).encode()})

        user_article_basic.foreachPartition(save_content_filter_history_to_recall)

if __name__ == '__main__':
    ur = UpdateRecall(500)
    ur.update_als_recall()
    ur.update_content_recall()