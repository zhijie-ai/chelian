#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/11/12 8:57                         #
#                                              #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#-----------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('zhengqi_train.txt',delimiter='\t')
test = pd.read_csv('zhengqi_test.txt',delimiter='\t')
sns.distplot(train.V0,color='r',label='train')
sns.distplot(test.V0,color='b',label='test')
plt.show()