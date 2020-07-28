#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/8/5 21:43                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
LABEL = 10

y_samples = np.zeros([10,LABEL])
for i in range(LABEL):
    for j in range(LABEL):
        y_samples[i*LABEL+j,i] = 1

print(y_samples)