#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/11/25 13:43                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import numpy as np
import pandas as pd

np.random.seed(1126)

data = np.random.randn(10,200)
data = np.array([list(np.exp(d)/np.sum(np.exp(d))) for d in data])
df = pd.DataFrame(data.T)# (200,10)
ranks = (df.values>np.diag(df)).sum(axis=0)
# print(ranks,df.values.shape,np.diag(df).shape)

data = np.arange(18).reshape(6,3)
print(data)
print(np.diag(data))
rank = (data>np.diag(data))
print(rank)