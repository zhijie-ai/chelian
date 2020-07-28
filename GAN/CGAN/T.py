#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'xiaojie'                        #
# CreateTime:                                 #
#       2019/4/29 14:18                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
LABEL=10
y_samples = np.zeros([100,LABEL])
for i in range(LABEL):
    for j in range(LABEL):
        y_samples[i*LABEL+j,i] = 1
# print(y_samples)

class T():
    def __init__(self,name,age):
        self.name = name
        self._age = age

    def __call__(self):
        print(self.name)

if __name__ == '__main__':
    t = T('xiaojie',10)
    print(t._age)
    print(t.name)
    print(t.__call__())
    print(callable(t))