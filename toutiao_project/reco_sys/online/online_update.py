#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/4 21:59                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))
from online import stream_sc,SIMILAR_DS,HOT_DS,NEW_ARTICLE_DS
from setting.default import DefaultConfig
import json
import time
from datetime import datetime
import logging
import redis

logger = logging.getLogger('online')

class OnlineRecall():
    '''
    在线计算部分
    1. 在线内容召回，实时写入用户点击或者操作文章的相似文章
    2. 在线新文章的召回
    3. 在线热门文章的召回
    '''
    def __init__(self):
        self.client = redis.StrictRedis(host=DefaultConfig.REDIS_HOST,
                                        port=DefaultConfig.REDIS_PORT,db=10)


    def _update_content_recall(self):
        '''
        在线内容召回计算
        :return:
        '''
        # {"actionTime":"2019-04-10 21:04:39","readTime":"","channelId":18,
        # "param":{"action": "click", "userId": "2", "articleId": "116644", "algorithmCombine": "C2"}}
        # x [,'json.....']
        def get_similar_online_recall(rdd):
            """
            解析rdd中的内容，然后进行获取计算
            :param rdd:
            :return:
            """
            # rdd-->数据本身
            # [row(1,2,3),row(4,5,6)] -->[[1,2,3],[4,5,6]]
            import happybase
            # 初始化happybase的链接
            pool = happybase.ConnectionPool(size=10,host='192.168.19.137',port=9090)
            with pool.connection() as conn:
                sim_table = conn.table('article_similar')
                for data in rdd.collect():
                    # 进行data字典处理过滤
                    if data['param']['action'] in ['click','collect','share']:
                        # 读取param当中articleid，相似的文章
                        # 根据用户点击流日志涉及文章找出与之最相似文章(基于内容的相似)，选取TOP-k相似的作为召回推荐结果
                        _dic = sim_table.row(str(data["param"]["articleId"]).encode(), columns=[b"similar"])
                        _srt = sorted(_dic.items(), key=lambda obj: obj[1], reverse=True)  # 按相似度排序
                        if _srt:
                            topKSimIds = [int(i[0].split(b":")[1]) for i in _srt[:10]]

                            # 根据历史推荐集过滤，已经给用户推荐过的文章
                            history_table = conn.table("history_recall")
                            _history_data = history_table.cells(
                                b"reco:his:%s" % data["param"]["userId"].encode(),
                                b"channel:%d" % data["channelId"]
                            )

                            history = []
                            if len(_history_data)>1:
                                for l in _history_data:
                                    history.extend(l)


                            # 根据历史召回记录，过滤召回结果
                            recall_list = list(set(topKSimIds)-set(history))

                            # 如果有推荐结果集，那么将数据添加到cb_recall表中，同时记录到历史记录表中
                            if recall_list:
                                recall_table = conn.table('cb_recall')
                                recall_table.put(
                                    b'recall:use:%s' %data['param']['userId'].encode(),
                                    {b'online:%d'%data['channelId']:str(recall_list).encode()}
                                )

                                history_table.put(
                                    b'reco:his:%s'%data['param']['userId'].encode(),
                                    {b'channelId:%d'%data['channelId']:str(recall_list).encode()}
                                )


        SIMILAR_DS.map(lambda x:json.loads(x[1])).foreachRDD(get_similar_online_recall)


    def _update_hot_redis(self):
        '''
        收集用户行为，更新热门文章分数
        :return:
        '''
        client = self.client
        # {"actionTime":"2019-04-10 21:04:39","readTime":"","channelId":18,
        # "param":{"action": "click", "userId": "2", "articleId": "116644", "algorithmCombine": "C2"}}

        def updateHotArticle(rdd):
            for data in rdd.collect():
                logger.info("{}, INFO: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), data))
                # 判断用户操作行为
                if data['param']['action'] in ['exposure','read']:
                    pass
                else:
                    client.zincrby('ch:{}:hot'.format(data['channelId']),1,data['param']['articleId'])

        HOT_DS.map(lambda x: json.loads(x[1])).foreachRDD(updateHotArticle)

    def _update_new_redis(self):
        '''
        更新频道新文章
        :return:
        '''
        client = self.client

        def computeFunction(rdd):
            for row in rdd.collect():
                channel_id,article_id = row.split(',')
                logger.info("{}, INFO: get kafka new_article each data:channel_id{}, article_id{}".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), channel_id, article_id))
                client.zadd('ch:{}:new'.format(channel_id),{article_id:time.time()})

        NEW_ARTICLE_DS.map(lambda x:x[1]).foreachRDD(computeFunction)

if __name__ == '__main__':
    import setting.logging as lg
    lg.create_logger()
    ore = OnlineRecall()
    ore._update_content_recall()
    ore._update_hot_redis()
    ore._update_new_redis()
    stream_sc.start()
    # 使用 ctrl+c 可以退出服务
    _ONE_DAY_IN_SECONDS = 60*60*24
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        pass

