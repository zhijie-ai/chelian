# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2022/3/18 15:09                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
X = np.random.randn(5,100)

# plt.imshow(X)
plt.imshow(np.transpose(X),
           cmap='bwr',
           aspect='auto',
           vmin=-2,
           vmax=2)
plt.show()