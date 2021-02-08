#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/7 22:35                       #
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
import hashlib
from setting.default import RAParam
from server.utils import HBaseUtils
from server.recall_service import ReadRecall
from server.redis_cache import get_cache_from_redis_hbase
from server.sort_service import lr_sort_service
from server import pool
from datetime import datetime
import logging
import json

logger = logging.getLogger('recommend')

sort_dict = {
    "LR": lr_sort_service,
    # "WDL":
}


def add_track(res, temp):
    """
    封装埋点参数
    :param res: 推荐文章id列表
    :param cb: 合并参数
    :param rpc_param: rpc参数
    :return: 埋点参数
        文章列表参数
        单文章参数
    """
    # 添加埋点参数
    track = {}

    # 准备曝光参数
    # 全部字符串形式提供，在hive端不会解析问题
    _exposure = {"action": "exposure", "userId": temp.user_id, "articleId": json.dumps(res),
                 "algorithmCombine": temp.algo}

    track['param'] = json.dumps(_exposure)
    track['recommends'] = []

    # 准备其它点击参数
    for _id in res:
        # 构造字典
        _dic = {}
        _dic['article_id'] = _id
        _dic['param'] = {}

        # 准备click参数
        _p = {"action": "click", "userId": temp.user_id, "articleId": str(_id),
              "algorithmCombine": temp.algo}

        _dic['param']['click'] = json.dumps(_p)
        # 准备collect参数
        _p["action"] = 'collect'
        _dic['param']['collect'] = json.dumps(_p)
        # 准备share参数
        _p["action"] = 'share'
        _dic['param']['share'] = json.dumps(_p)
        # 准备detentionTime参数
        _p["action"] = 'read'
        _dic['param']['read'] = json.dumps(_p)

        track['recommends'].append(_dic)

    track['timestamp'] = temp.time_stamp
    return track

class RecoCenter():
    '''
    推荐中心
    1. 处理时间戳逻辑
    2. 召回，排序，缓存
    '''
    def __init__(self):
        self.hbu = HBaseUtils(pool)
        self.recall_service = ReadRecall()

    def feed_recommend_time_stamp_logic(self,temp):
        '''
        用户刷新时间戳逻辑
        :param temp:ABTest传入的用户请求参数
        :return:
        '''
        # 1. 获取用户的历史数据库中最近一次的时间戳lt
        # 如果用户没有过历史记录
        try:
            last_stamp = self.hbu.get_table_row('history_recommend',
                                               'reco:his:{}'.format(temp.user_id).encode(),
                                               'channel:{}'.format(temp.channel_id).encode(),
                                               include_timestamp=True)[1]
            logger.info("{} INFO get user_id:{} channel:{} history last_stamp".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))
        except Exception as e:
            logger.info("{} INFO get user_id:{} channel:{} history last_stamp, exception:{}".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id, e))
            last_stamp = 0

        # 2. 如果lt<用户请求时间戳用户的刷新操作
        if last_stamp < temp.time_stamp:
            # 走正常的推荐流程
            # 缓存读取，召回排序流程

            # last_stamp应该是temp.time_stamp前面的一条数据
            # 返回给用户上一条时间戳给定义为last_stamp
            res = get_cache_from_redis_hbase(temp,self.hbu)

            if not res:
                logger.info("{} INFO cache is null get user_id:{} channel:{} recall/sort data".
                            format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))
                res = self.user_reco_list(temp)

            temp.time_stamp = last_stamp
            _track = add_track(res,temp)
            # 读取用户召回结果返回

        else :
            logger.info("{} INFO read user_id:{} channel:{} history recommend data".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))
            # 3、如果lt >= 用户请求时间戳, 用户才翻历史记录
            # 根据用户传入的时间戳请求，去读取对应的历史记录
            # temp.time_stamp
            # 1559148615353,hbase取出1559148615353小的时间戳的数据， 1559148615354
            try:
                row = self.hbu.get_table_cells('history_recommend',
                                               'reco:his:{}'.format(temp.user_id).encode(),
                                               'channel:{}'.format(temp.channel_id).encode(),
                                               timestamp=temp.time_stamp + 1,
                                               include_timestamp=True)
            except Exception as e:
                logger.warning("{} WARN read history recommend exception:{}".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e))
                row = []
                res = []

            # [(,), ()]
            # 1559148615353, [15140, 16421, 19494, 14381, 17966]
            # 1558236647437, [18904, 14300, 44412, 18238, 18103, 43986, 44339, 17454, 14899, 18335]
            # 1558236629309, [43997, 14299, 17632, 17120]

            # 3步判断逻辑
            # 1. 如果没有历史数据，则返回时间戳0以及空列表
            # 1558236629307
            if not row:
                temp.time_stamp = 0
                res=[]
            elif len(row) == 1 and row[0][1] == temp.time_stamp:
                # [([43997, 14299, 17632, 17120], 1558236629309)]
                # 2、如果历史数据只有一条，返回这一条历史数据以及时间戳正好为请求时间戳，修改时间戳为0，表示后面请求以后就没有历史数据了(APP的行为就是翻历史记录停止了)
                res = row[0][0]
                temp.time_stamp=0
            elif len(row)>=2:
                res =row[0][0]
                temp.time_stamp = int(row[1][1])
                # 3、如果历史数据多条，返回最近的第一条历史数据，然后返回之后第二条历史数据的时间戳

            # res bytes ->list
            # list str->int id
            res = list(map(int,eval(res)))

            logger.info(
                "{} INFO history:{}, {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), res, temp.time_stamp))

            _track = add_track(res, temp)
            _track['param']=''

        return _track

    def user_reco_list(self,temp):
        """
        用户的下拉刷新获取新数据的逻辑
        - 1、循环算法组合参数，遍历不同召回结果进行过滤
        - 2、过滤当前该请求频道推荐历史结果，如果不是0频道需要过滤0频道推荐结果，防止出现
        - 3、过滤之后，推荐出去指定个数的文章列表，写入历史记录，剩下多的写入待推荐结果
        :return:
        """
        # 1. 循环算法组合参数，遍历不同召回结果进行过滤
        reco_set = []
        # (1, [100, 101, 102, 103, 104], [])
        for number in RAParam.COMBINE[temp.algo[1]]:
            if number ==103:
                _res = self.recall_service.read_redis_new_article(temp.channel_id)
                reco_set = list(set(reco_set).union(set(_res)))
            elif number==104:
                _res  = self.recall_service.read_redis_hot_article(temp.channel_id)
                reco_set = list(set(reco_set).union(set(_res)))
            else:
                _res = self.recall_service.read_hbase_recall(RAParam.RECALL[number][0],
                                                             'recall:user:{}'.format(temp.user_id).encode(),
                                                             '{}:{}'.format(RAParam.RECALL[number][1],
                                                                            temp.channel_id).encode())
                reco_set = list(set(reco_set).union(set(_res)))

        # - 2、过滤当前该请求频道推荐历史结果，如果不是0频道需要过滤0频道推荐结果，防止出现
        history_list = []
        try:
            data = self.hbu.get_table_cells('history_recommend',
                                            'reco:his:{}'.format(temp.user_id).encode(),
                                            'channel:{}'.format(temp.channel_id).encode())

            for _ in data:
                history_list = list(set(history_list).union(set(eval(_))))

            logger.info("{} INFO read user_id:{} channel_id:{} history data".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))
        except Exception as e:
            # 打印日志
            logger.warning(
                "{} WARN filter history article exception:{}".format(datetime.now().
                                                                     strftime('%Y-%m-%d %H:%M:%S'), e))

        try:
            #0频道
            data = self.hbu.get_table_cells('history_recommend',
                                            'reco:his:{}'.format(temp.user_id).encode(),
                                            'channel:{}'.format(0).encode())

            for _ in data:
                history_list = list(set(history_list).union(set(eval(_))))

            logger.info("{} INFO read user_id:{} channel_id:{} history data".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, 0))

        except Exception as e:
            # 打印日志
            logger.warning(
                "{} WARN filter history article exception:{}".format(datetime.now().
                                                                     strftime('%Y-%m-%d %H:%M:%S'), e))

        # reco_set  history_list
        # # - 3、过滤之后，推荐出去指定个数的文章列表，写入历史记录，剩下多的写入待推荐结果
        reco_set = list(set(reco_set).difference(set(history_list)))
        logger.info(
            "{} INFO after filter history:{}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), reco_set))

        # 模型对召回的结果排序
        # temp.user_id,reco_set
        _sort_num = RAParam.COMBINE[temp.algo][2][0]
        # 'LR'
        reco_set = sort_dict[RAParam.SORT[_sort_num]](reco_set,temp,self.hbu)

        if not reco_set:
            return reco_set
        else:
            # 如果reco_set小于用户需要推荐的文章12
            if len(reco_set)<=temp.article_num:
                res = reco_set
            else:
                # 大于要推荐的文章结果
                res = reco_set[:temp.article_num]

                # # 将剩下的文章列表写入待推荐的结果
                self.hbu.get_table_put('wait_recommend',
                                       'reco:{}'.format(temp.user_id).encode(),
                                       'channel:{}'.format(temp.channel_id).encode(),
                                       str(reco_set[temp.article_num:]).encode(),
                                       timestamp=temp.time_stamp)

                # 直接写入历史记录当中，表示这次又成功推荐一次
                self.hbu.get_table_put('history_recommend',
                                       'reco:his:{}'.format(temp.user_id).encode(),
                                       'channel:{}'.format(temp.channel_id).encode(),
                                       str(res).encode(),
                                       timestamp=temp.time_stamp)

                return res


