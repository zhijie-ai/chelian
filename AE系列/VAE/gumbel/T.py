# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/3/18 11:16                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot   as plt
# matplotlib.use('Agg')

def softmax(logits):
    max_value = np.max(logits)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp)
    dist = exp / exp_sum
    return dist

def generalized_softmax(logits, temperature=1):
    logits = logits / temperature
    return softmax(logits)

np.random.seed(1111)
n = 10
logits = (np.random.random(n) - 0.5) * 2  # (-1, 1)
x = range(n)
plt.subplot(1, 3, 1)
t = .1
plt.bar(x, generalized_softmax(logits, t))
plt.show()