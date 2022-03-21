# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/3/19 16:30                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def gumbel_sample(logits, num):
    n = []
    for i in range(num):
        u = np.random.rand(len(logits))
        l = (logits + (-np.log(-np.log(u))))/0.01
        p = np.exp(l)/np.sum(np.exp(l))
        n.append(np.argmax(p))

    return n

logits = np.random.random(10)
p = np.exp(logits)/np.sum(np.exp(logits))

sample1 = np.random.choice(len(logits), p=p,size=1000)
sample2 = gumbel_sample(logits, 1000)
plt.subplot(1, 2, 1)
plt.hist(sample1)

plt.subplot(1, 2, 2)
plt.hist(sample2)
plt.show()
