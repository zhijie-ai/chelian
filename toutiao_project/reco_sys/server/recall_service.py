#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/8 16:40                       #
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

from reco_sys.server import redis_client
from reco_sys.server import pool
import logging
from datetime import datetime
from reco_sys.server.utils import HBaseUtils

logger = logging.getLogger('recommend')


class ReadRecall():
    '''
    读取召回集的结果
    '''
    def __init__(self):
        self.client = redis_client
        self.hbu = HBaseUtils(pool)
        self.hot_num = 10

    def read_hbase_recall(self,table_name,key_format,column_format):
        '''
        读取用户的召回集结果
        :param table_name:
        :param key_format:
        :param column_format:
        :return:
        '''
        reco_set = []
        try:
            data = self.hbu.get_table_cells(table_name,key_format,column_format)
            for _ in data:
                reco_set = list(set(reco_set).union(set(eval(_))))
            # 删除这个召回结果
            self.hbu.get_table_delete(table_name,key_format,column_format)
        except Exception as e:
            logger.warning("{} WARN read {} recall exception:{}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                        table_name, e))
        return reco_set

    def read_redis_new_article(self, channel_id):
        """
        读取用户的新文章
        :param channel_id:
        :return:
        """
        logger.warning("{} INFO read channel {} redis new article".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                          channel_id))
        key = 'ch:{}:new'.format(channel_id)
        try:

            reco_list = self.client.zrevrange(key, 0, -1)

        except Exception as e:
            logger.warning(
                "{} WARN read new article exception:{}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e))

            reco_list = []

        return list(map(int, reco_list))

    def read_redis_hot_article(self, channel_id):
        """
        读取新闻章召回结果
        :param channel_id: 提供频道
        :return:
        """
        logger.warning("{} INFO read channel {} redis hot article".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                          channel_id))
        _key = "ch:{}:hot".format(channel_id)
        try:
            res = self.client.zrevrange(_key, 0, -1)

        except Exception as e:
            logger.warning(
                "{} WARN read new article exception:{}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e))
            res = []

        # 由于每个频道的热门文章有很多，因为保留文章点击次数
        res = list(map(int, res))
        if len(res) > self.hot_num:
            res = res[:self.hot_num]
        return res

if __name__ == '__main__':
    rr = ReadRecall()
    print(rr.read_hbase_recall('cb_recall',b'recall:user:1115629498121846784',b'als:18'))
    print(rr.read_redis_new_article(18))