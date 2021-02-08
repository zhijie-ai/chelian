#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/1/30 20:22                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 增量更新用户画像
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))
from offline import SparkSessionBase
import time

class UpdateUserProfile(SparkSessionBase):
    '''
    离线相关定时处理
    '''
    SPARK_APP_NAME='updateUser'
    ENABLE_HIVE_SUPPORT = True

    SPARK_EXECUTOR_MEMORY='7g'

    def __init__(self):
        self.spark = self._create_spark_session()

        # 用户文章行为中间表结构
        self._user_article_basic_column = ['user_id','action_time',
                                           'article_id','channel_id','shared','clicked',
                                           'collected','exposure','read_time']

    def update_user_action_basic(self):
        """
        更新用户行为日志到用户行为基础标签表
        :return:
        """
        # 读取日志数据进行处理得到基础数据表
        self.spark.sql('use profile')
        # 如果hadoop没有今天该日期文件，则没有日志数据，结束
        time_str = time.strftime("%Y-%m-%d", time.localtime())
        sqlDF = self.spark.sql(
            'select actionTime,readTime,channelId,param.articleId,param.algorithmCombine,param.action,param.userId from user_action where dt={}'.format(time_str)
        )

        if sqlDF.collect():
            def _compute(row):
                _list=[]
                if row.action =='exposure':
                    for article_id in eval(row.articleId):
                        _list.append([
                            row.userId,row.actionTime,article_id,row.channelId,False,False,False,True,row.readTime
                        ])
                    return _list
                else:
                    class Temp():
                        shared=False
                        clicked=False
                        collected=False
                        read_time=''

                    _tp = Temp()

                    if row.action=='share':
                        _tp.shared=True
                    elif row.action=='click':
                        _tp.clicked=True
                    elif row.action=='collect':
                        _tp.collected=True

                    elif row.action == 'read':
                        _tp.clicked=True
                    else:
                        pass

                    _list.append([
                        row.userId,row.actionTime,int(row.articleId),row.channelId,_tp.shared,_tp.clicked,
                        _tp.collected,True,row.readTime
                    ])
                    return _list

                # 查询内容，将原始日志进行处理
                _res = sqlDF.rdd.flatMap(_compute)
                data = _res.toDF(self._user_article_basic_column)

                # 将得到的每个用户对每篇文章的结果合并,写入到用户与文章的基础行为信息表中
                old = self.spark.sql('select * from user_article_basic')
                old.unionAll(data).retisterTempTable('temptable')
                # 直接写入用户与文章基础行为表
                self.spark.sql(
                    'inser overwrite table user_article_basic select user_id,max(action_time) as action_time,article_id,max(channel_id) channel_id,max(shared) shared ,max(clicked) as clicked,max(collected) as collected,max(exposure) as exposure,max(read_time) as read_time from temptable group by user_id,article_id'
                )
                return True
            return False
        else:
            return False


    def update_user_label(self):
        # 查询用户与文章基础行为信息表结果，合并每个频道的主题词中间表，与中间表关联
        # result = self.spark.sql('select * from user_article_basic order by article_id')
        # 对每个主题下的主题词进行合并

        # 制作每个人的字典对应每个标签的分值，计算每个用户的变迁权重公式
        user_article_ = self.spark.sql('select * from user_article_basic').drop('channel_id')
        self.spark.sql('use article')
        article_label = self.spark.sql('select article_id,channel_id,topics from article_profile')
        click_article_res = user_article_.join(article_label,how='left',on=['article_id'])

        # 将字段的列表爆炸
        import pyspark.sql.functions as F
        click_article_res = click_article_res.withColumn('topic',F.explode('topics')).drop('topics')

        # 计算每个用户对每篇文章的标签的权重
        def compute_weights(rowpartiton):
            weightsOfaction = {
                "read_min": 1,
                "read_middle": 2,
                "collect": 2,
                "share": 3,
                "click": 5
            }

            from datetime import  datetime
            import numpy as np
            import json
            import happybase

            pool = happybase.ConnectionPool(size=10,host='192.168.19.137',port=9090)

            # 读取文章的标签数据
            # 计算权重值
            # 时间间隔
            for row in rowpartiton:
                t = datetime.now()-datetime.strptime(row.action_time,'%Y-%m-%d %H:%M:%S')
                # 时间衰减系数
                time_exp = 1/(np.log(t.days+1)+1)

                if row.read_time=='':
                    r_t = 0
                else:
                    r_t = int(row.read_time)

                # 浏览时间分数
                is_read = weightsOfaction['read_middle'] if r_t>1000 else weightsOfaction['read_min']

                weights=time_exp*(
                    row.shared*weightsOfaction['share']+row.collected*weightsOfaction['collect']+ \
                    row.clicked*weightsOfaction['click']+is_read
                )

                with pool.connection() as conn:
                    table=conn.table('user_profile')
                    table.put('user:{}'.format(row.user_id).encode(),
                              {'partial:{}:{}'.format(row.channel_id,row.topic).encode():json.dumps(weights).encode()})


        click_article_res.foreachPartition(compute_weights)

    def update_user_info(self):
        '''
        更新用户的基础信息画像
        :return:
        '''
        self.spark.sql('use toutiao')

        user_basic = self.spark.sql('select user_id,gender,birthday from user_profile')

        # 更新用户基本信息
        def _update_user_basic(partition):

            from datetime import  datetime
            from datetime import  date
            import json
            import happybase

            pool = happybase.ConnectionPool(size=10, host='192.168.19.137', port=9090)
            for row in partition:
                age = 0
                if row.birthday!='null':
                    born = datetime.strptime(row.birthday,'%Y-%m-%d')
                    today = date.today()
                    age = today.year-born.year-((today.month,today.day)<(born.month,born.day))

                with pool.connection() as conn:
                    table = conn.table('user_profile')
                    table.put('user:{}'.format(row.user_id).encode(),
                              {'basic:gender'.encode():json.dumps(row.gender).encode()})
                    table.put('user:{}'.format(row.user_id).encode(),
                              {'basic:birthday'.encode():json.dumps(age).encode()})

        user_basic.foreachPartition(_update_user_basic)



if __name__ == '__main__':
    op = UpdateUserProfile()
    op.update_user_action_basic()
    op.update_user_label()
    op.update_user_info()