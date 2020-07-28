#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2019/11/26 20:34                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import tensorflow as tf
import glob
import sklearn
import pandas as pd
import numpy as np

COLUMNS = ['label', 'browser', 'city', 'creativeid', 'iid_clked', \
           'iid_imp', 'itemtype', 'network', 'operation', 'os', \
           'province', 'psid_abs', 'pvday', 'pvhour', 'read', 'sessionid', 'source', 'userid']