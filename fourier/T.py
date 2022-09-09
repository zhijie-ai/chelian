# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/8/24 9:48                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(-2*np.pi, 2*np.pi, 1000)
f = np.sin(3*x) + np.sin(5*x)
w = 1
n = 20000
f2 = 0
for i in range(1,n+1):
    k = 2*i -1
    f2+= np.sin(k*w*x)/k

plt.plot(x, f2)
plt.grid()
plt.show()