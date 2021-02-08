#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/23 11:32                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
# 双向seq2seq理论分析:https://mp.weixin.qq.com/s/tziT9OE2qxJX0pIXr6fOIA
# https://github.com/bojone/seq2seq/blob/master/seq2seq_bidecoder.py

import numpy as np
from tqdm import tqdm
import os,json
from keras.layers import *
# from keras_layer_normalization import LayerNormalization
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam