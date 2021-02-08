#-----------------------------------------------
# -*- encoding=utf-8 -*-                       #
# __author__:'焉知飞鱼'                         #
# CreateTime:                                  #
#       2020/11/12 8:47                         #
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

cols = 8
rows = 5
i = 1
plt.figure(figsize=(80,30))
for col in test.columns:
    ax = plt.subplot(rows,cols,i)
    ax = sns.distplot(train[col],color='Red',kde=True,label='train')
    ax = sns.distplot(test[col],color='Blue',kde=True,label='test')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    i+=1

plt.legend()
# plt.show()
plt.savefig('show.jpg')