#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/7/22 14:31                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
# plt.switch_backend('agg')

plt.plot(np.arange(100),color='r')
plt.plot(np.arange(100,199),color='g')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['r','g'],loc='best')
# plt.savefig('1.jpg')
plt.show()
plt.close()