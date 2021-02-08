#----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2020/4/29 16:48                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
#----------------------------------------------
import numpy as np
import time

data = np.random.randint(0,100,20)
# data = [1,2,4,6,6,5,8]
print(data)

def quick_sort(data,l,r):
    if l<r:
        i = l
        j = r
        x = data[i]
        while i<j:
            while(i<j and data[j]>x):#从右边开始找出一个比x小的数
                j-=1
            if i<j:
                data[i] = data[j]
                i+=1
            while(i<j and data[i]<x):
                i+=1
            if i<j:
                data[j]=data[i]
                j-=1

        data[i] = x
        quick_sort(data,l,i-1)
        quick_sort(data,i+1,r)

quick_sort(data,0,len(data)-1)
print(data)
print(type(data))

def quick_sort2(seq):
    if len(seq) <2:
        return seq

    else:
        base = seq[0]
        left = [elem for elem in seq[1:] if elem <=base]
        right = [elem for elem in seq[1:] if elem >base]
        return quick_sort2(left) + [base]  + quick_sort2(right)

data = quick_sort2(data)
print(data)
print(len(data))
print(type(data))


