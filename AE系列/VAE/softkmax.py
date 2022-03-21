# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/3/18 18:24                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import numpy as np

np.random.seed(1127)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

x = np.random.randint(1,60,10)
print(x, np.sort(x))
p = np.zeros(len(x))
x = x-np.min(x)
for i in range(5):
    y = x-p*x
    p = p+softmax(y)
    print(p)