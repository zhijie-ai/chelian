#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/29 15:50                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import random
import numpy as np
import time

data = np.random.randint(0,10000000,100000)
# data = np.random.randint(0,100,20)
# data = [1,2,4,6,6,5,8]
print(data[0:20])


def bubble_sort(data):
    for i in range(0,len(data))[::-1]:
        flag=0
        for j in range(0,i):
            if data[j]>data[j+1]:
                temp=data[j]
                data[j]=data[j+1]
                data[j+1]=temp
                flag=1

        if not flag:
            break

t1 = time.time()
bubble_sort(data)
print(data[0:20])
t2 = time.time()
print('time cost:{} s'.format((t2-t1)))