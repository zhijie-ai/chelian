#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/7/13 11:50                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import utils as ut


workdir = 'MQ2008-semi'

query_url_feature, query_url_index, query_index_url = \
    ut.load_all_query_url_feature(workdir + '/Large_norm.txt', 46)

query_pos_train = ut.get_query_pos(workdir + '/train.txt') # rank>0

