#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/1 17:07                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------

import numpy as np

def make_one_hot(indices,size):
    as_one_hot = np.zeros((indices.shape[0],size))
    as_one_hot[np.arange(0,indices.shape[0]),indices] = 1.0
    return as_one_hot