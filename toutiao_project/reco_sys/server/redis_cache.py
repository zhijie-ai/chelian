#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/9 17:08                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 推荐系统缓存的读取
# 首先从redis一级缓存读取，如果有，则直接返回，如果没有，则读取wait_recommend hbase二级缓存表
# 如果hbase二级缓存表中的数据大于100，则将前100条写入redis一级缓存，从100条开始，写入二级缓存。然后从一级缓存redis
# 中取数据，并删除redis中已经取出的数据
from server import cache_client
import logging
from datetime import datetime

logger = logging.getLogger('recommend')

def get_cache_from_redis_hbase(temp,hbu):
    '''
    读取用户的缓存结果
    :param temp: 用户的请求参数
    :param hbu:hbase读取
    :return:
    '''

    # 1. redis 8号库的读取
    key = 'reco:{}:{}:art'.format(temp.user_id,temp.channel_id)
    res = cache_client.zrevrange(key,0,temp.article_num-1)
    if res:
        cache_client.zrem(key,*res)

    else:
        # 2. redis中没有数据，进行wait_recommend读取，放入redis中
        cache_client.delete(key)

        try:
            hbase_cache = eval(hbu.get_table_row('wait_recommmend',
                                                 'reco:{}'.format(temp.user_id).encode(),
                                                 'channel:{}'.format(temp.channel_id).encode()))
        except Exception as e:
            logger.warning("{} WARN read user_id:{} wait_recommend exception:{} not exist".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, e))

            hbase_cache=[]

        # 推荐出去的结果放入历史结果
        if not hbase_cache:
            return hbase_cache

        if len(hbase_cache)>100:
            logger.info(
                "{} INFO reduce cache  user_id:{} channel:{} wait_recommend data".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))

            redis_cache = hbase_cache[:100]

            cache_client.zadd(key,dict(zip(redis_cache,range(len(redis_cache)))))
            hbu.get_table_put('wait_recommend',
                              'reco:{}'.format(temp.user_id).encode(),
                              'channel:{}'.format(temp.channel_id).encode(),
                              str(hbase_cache[100:]).encode(),
                              timestamp=temp.time_stamp)
        else:
            logger.info(
                "{} INFO delete user_id:{} channel:{} wait_recommend cache data".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))

            cache_client.zadd(key,dict(zip(hbase_cache,range(len(hbase_cache)))))
            hbu.get_table_put('wait_recommend',
                              'reco:{}'.format(temp.user_id).encode(),
                              'channel_id:{}'.format(temp.channel_id).encode(),
                              str([]).encode(),timestamp=temp.time_stamp)

        res = cache_client.zrevrange(key,0,temp.article_num-1)
        if res :
            cache_client.zrem(key,*res)

    # 存进历史推荐表中
    res = list(map(int,res))
    logger.info("{} INFO get cache data and store user_id:{} channel:{} cache data".format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))

    # 直接写入历史记录当中，表示这次又成功推荐一次
    hbu.get_table_put('history_recommend',
                      'reco:his:{}'.format(temp.user_id).encode(),
                      'channel:{}'.format(temp.channel_id).encode(),
                      str(res).encode(),
                      timestamp=temp.time_stamp)

    return res