#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/2/7 22:15                       #
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
from abtest import user_reco_pb2_grpc
from abtest import user_reco_pb2
import grpc
from setting.default import DefaultConfig
import time

def test():
    article_dict={}
    # 构造传入数据

    req_article = user_reco_pb2.User()
    req_article.user_id = '1115629498121846784'
    req_article.channel_id = 18
    req_article.article_num = 10
    req_article.time_stamp = int(time.time()*1000)

    with grpc.insecure_channel(DefaultConfig.RPC_SERVER) as rpc_cli:
        try:
            stub = user_reco_pb2_grpc.UserRecommendStub(rpc_cli)
            resp = stub.user_recommend(req_article)
        except Exception as e:
            print(e)
            article_dict['param']=[]
        else:

            # 解析返回结果参数
            article_dict['exposure_param'] = resp.exposure

            reco_arts = resp.recommends
            reco_art_param=[]
            reco_list=[]
            for art in reco_arts:
                reco_art_param.append({'article_id':art.article_id,
                                       'params':{'click':art.params.click,
                                                 'collect':art.params.collect,
                                                 'share':art.params.share,
                                                 'read':art.params.read}})
                reco_list.append(art.article_id)

            article_dict['param'] = reco_art_param

            # 文章列表以及参数（曝光参数 以及 每篇文章的点击等参数）
            print(reco_list, article_dict)

if __name__ == '__main__':
    test()
