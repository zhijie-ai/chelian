#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/26 10:54                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import pandas as pd

import numpy as np

import random

train_data = np.zeros([3,10000],dtype=np.int32)

test_data = np.zeros([3,200],dtype=np.int32)
random.seed(1)

for i in range(10000):
    train_data[0,i] = random.randint(0,2000)
    train_data[1,i] = random.randint(0,2000)
    train_data[2,i] = random.randint(0,20000)

for i in range(200):
    test_data[0,i] = random.randint(0, 40)
    test_data[1,i] = random.randint(0, 20)
    test_data[2,i] = random.randint(0, 2000)

train_data = np.transpose(train_data)
test_data = np.transpose(test_data)


# train_df = pd.DataFrame(train_data,columns=['SessionId','ItemId','Timestamps']).to_csv('data/train.csv')
test_df = pd.DataFrame(test_data,columns=['SessionId','ItemId','Timestamps']).to_csv('data/test.csv')